# app.py

import stats_tests
import test_model

def run_stats_tests():
    stats_tests.main()

def run_test_model():
    test_model.main()

def main():
    run_stats_tests()
    run_test_model()

if __name__ == "__main__":
    main()