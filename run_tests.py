# run_tests.py
import unittest
import sys
import os
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

def main():
    print('=' * 70)
    print('STUDENT PERFORMANCE ANALYTICS PLATFORM - RELIABILITY & EVALUATION SUITE')
    print('=' * 70)
    print(f'Test Directory: {os.path.join(_CURRENT_DIR, "tests")}')
    print('Suites: ML Correctness, Evaluation Metrics, AI Agent, API Endpoints')
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_files = [
        'test_ml_correctness',
        'test_evaluation_metrics',
        'test_ai_agent',
        'test_api_endpoints'
    ]

    for tf in test_files:
        try:
            import importlib.util
            fpath = os.path.join(_CURRENT_DIR, 'tests', f'{tf}.py')
            spec = importlib.util.spec_from_file_location(tf, fpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[tf] = mod
            spec.loader.exec_module(mod)
            sub_suite = loader.loadTestsFromModule(mod)
            suite.addTests(sub_suite)
            print(f'  [LOADED] tests/{tf}.py ({sub_suite.countTestCases()} test cases)')
        except Exception as e:
            print(f'  [ERROR] Failed loading {tf}: {e}')

    print()
    print('-' * 70)
    print('Executing Test Suite...')
    print('-' * 70)

    start_time = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print()
    print('=' * 70)
    print('TEST SUITE EXECUTION SUMMARY')
    print('=' * 70)
    print(f'Total Tests Run: {result.testsRun}')
    print(f'Successes:      {result.testsRun - len(result.failures) - len(result.errors)}')
    print(f'Failures:       {len(result.failures)}')
    print(f'Errors:         {len(result.errors)}')
    print(f'Total Duration: {elapsed:.3f} seconds')
    print('=' * 70)

    if result.wasSuccessful():
        print('>>> ALL RELIABILITY & EVALUATION TESTS PASSED SUCCESSFULLY! <<<')
        sys.exit(0)
    else:
        print('>>> SOME TESTS FAILED. PLEASE REVIEW LOGS ABOVE. <<<')
        sys.exit(1)

if __name__ == '__main__':
    main()
