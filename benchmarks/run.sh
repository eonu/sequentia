echo "sequentia"
python test_sequentia.py --n-jobs 16 --number 10
echo

echo "aeon"
python test_aeon.py --n-jobs 16 --number 10
echo

echo "tslearn"
python test_tslearn.py --n-jobs 16 --number 10
echo

echo "sktime"
python test_sktime.py --n-jobs 16 --number 10
echo

echo "pyts"
python test_pyts.py --n-jobs 16 --number 10
echo
