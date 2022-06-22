    
import traceback


def return_none_if_fail(is_debug=False):
    def _return_none_if_fail(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if is_debug:
                    print(e)
                    traceback.print_exc()
                return None

        return wrapper
    return _return_none_if_fail
