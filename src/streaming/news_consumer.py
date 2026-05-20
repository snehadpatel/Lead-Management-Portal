"""
Lightweight Kafka consumer for processing incoming news articles.
Consumes JSON articles from `news_raw`, performs sentiment analysis
via HuggingFace `transformers` (FinBERT) if available, and emits
summarized signals to `market_signals` topic or local JSONL fallback.

Usage:
  export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
  python -m src.streaming.news_consumer
"""
from __future__ import annotations

import os
import json
import time
from typing import Optional

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except Exception:
    KafkaConsumer = None
    KafkaProducer = None
    KAFKA_AVAILABLE = False


def analyze_text_with_finbert(text: str) -> tuple[str, float]:
    """Attempt to analyze sentiment using a transformers pipeline.
    Falls back to simple heuristic if transformers not installed or model fails.
    """
    try:
        from transformers import pipeline
        model = os.environ.get("FINBERT_MODEL", "yiyanghkust/finbert-tone")
        nlp = pipeline("sentiment-analysis", model=model, device=-1)
        out = nlp(text[:1000])[0]
        label = out.get("label")
        score = float(out.get("score", 0.0))
        return label, score
    except Exception:
        # Fallback heuristic
        text_l = (text or "").lower()
        pos = sum(1 for w in ("good bull growth buy up positive gain profit strong".split()) if w in text_l)
        neg = sum(1 for w in ("bad bear loss sell down negative drop risk crash weak".split()) if w in text_l)
        if pos > neg:
            return "positive", 0.8
        elif neg > pos:
            return "negative", 0.8
        else:
            return "neutral", 0.5


def run_consumer():
    topic_in = os.environ.get("KAFKA_NEWS_TOPIC", "news_raw")
    topic_out = os.environ.get("KAFKA_SIGNALS_TOPIC", "market_signals")

    if not KAFKA_AVAILABLE:
        print("⚠️ kafka-python not installed. Falling back to file-based processing.")
        consumer = None
        producer = None
    else:
        servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
        consumer = KafkaConsumer(topic_in, bootstrap_servers=servers, value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                 auto_offset_reset="latest", enable_auto_commit=True, consumer_timeout_ms=1000)
        producer = KafkaProducer(bootstrap_servers=servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    fallback_out = os.environ.get("STREAMING_FALLBACK_OUT", "streaming_output/market_signals.jsonl")
    os.makedirs(os.path.dirname(fallback_out), exist_ok=True)

    print(f"Starting news consumer (input={topic_in}, output={topic_out})")

    while True:
        try:
            if consumer:
                msgs = consumer.poll(timeout_ms=1000)
                # kafka-python poll returns dict of partitions
                records = []
                for tp, batch in msgs.items():
                    for msg in batch:
                        records.append(msg.value)
            else:
                # No Kafka — scan local dataset folder for new JSON files
                records = []
                data_dir = os.environ.get("LOCAL_NEWS_DIR", "datasets/unstructured/financial_news")
                for fn in os.listdir(data_dir)[:10]:
                    if fn.endswith('.json'):
                        path = os.path.join(data_dir, fn)
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                records.append(json.load(f))
                        except Exception:
                            continue

            for article in records:
                text = article.get('Body_Text') or article.get('Title') or ""
                label, conf = analyze_text_with_finbert(text)

                signal = {
                    "source": article.get('Source'),
                    "category": article.get('Category'),
                    "title": article.get('Title'),
                    "published_date": article.get('Published_Date'),
                    "sentiment": label,
                    "confidence": round(float(conf), 4),
                    "ingested_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                }

                if KAFKA_AVAILABLE and producer:
                    try:
                        producer.send(topic_out, signal)
                        producer.flush()
                        print(f"→ Emitted signal to Kafka: {signal['sentiment']} ({signal['confidence']}) - {signal['title'][:60]}")
                    except Exception as e:
                        print(f"⚠️ Failed to emit to Kafka: {e}")
                        with open(fallback_out, 'a', encoding='utf-8') as fo:
                            fo.write(json.dumps(signal) + "\n")
                else:
                    with open(fallback_out, 'a', encoding='utf-8') as fo:
                        fo.write(json.dumps(signal) + "\n")
                    print(f"→ Wrote signal to {fallback_out}: {signal['sentiment']} - {signal['title'][:60]}")

            # Sleep briefly when idle
            time.sleep(float(os.environ.get('CONSUMER_POLL_INTERVAL', '2.0')))

        except KeyboardInterrupt:
            print("Shutting down consumer")
            break
        except Exception as e:
            print(f"Consumer error: {e}")
            time.sleep(5.0)


if __name__ == '__main__':
    run_consumer()
