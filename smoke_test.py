import sys
import os

# Add src to PYTHONPATH
sys.path.append(os.path.abspath("src"))

try:
    from lume_platform.ml.buddy_engine import BuddyEngine
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def run_test():
    try:
        engine = BuddyEngine()
        
        # Investor query
        investor_query = "How do I start investing?"
        investor_reply = engine.generate(query=investor_query, audience="investor")
        print(f"Investor Query: {investor_query}")
        print(f"Response: {investor_reply.response[:100]}...")
        print(f"Confidence: {investor_reply.confidence}")
        print("-" * 20)
        
        # Distributor query
        distributor_query = "How do I onboard as a distributor?"
        distributor_reply = engine.generate(query=distributor_query, audience="distributor")
        print(f"Distributor Query: {distributor_query}")
        print(f"Response: {distributor_reply.response[:100]}...")
        print(f"Confidence: {distributor_reply.confidence}")
        
    except Exception as e:
        print(f"RuntimeError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
