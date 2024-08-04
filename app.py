# app.py

# we import stats_tests and test_model
import stats_tests
import test_model

# so that in the following 2 functions we can run those
# imported modules main functions.
def run_stats_tests():
    stats_tests.main()

def run_test_model():
    test_model.main()

# here in our own main function we call the local
# functions we have created
def main():
    run_stats_tests()
    run_test_model()

# this if-statement allows us to run our module directly.
# if this module is executed as the main program, 
# the main function will be executed.
if __name__ == "__main__":
    main()