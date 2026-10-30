import sys

def test_version_python():
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12