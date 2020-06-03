import pytest
from main import avg, check_video_or_pic

### avg()

def test_avg_empty_list():
    assert avg(list()) == 0

def test_avg_123():
    lst = [1, 2, 3]
    assert avg(lst) == 2

def test_avg_float():
    lst = [30.3592, 12.9384, 7.9384]
    res = avg(lst)
    assert res > 17.07 and res < 17.08

def test_avg_one_arg():
    lst = [12]
    assert avg(lst) == 12

## check_video_or_pic
def test_check_img():
    assert check_video_or_pic("test.jpg") == True

def test_check_vid():
    assert check_video_or_pic("test.mp4") == False

def test_check_exit():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        check_video_or_pic("test.xls")
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

