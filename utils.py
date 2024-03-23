def rand_str(n):
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))