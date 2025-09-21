# answer_keys.py
SUBJECTS = {
    'Python': (1, 20), 'EDA': (21, 40), 'SQL': (41, 60),
    'Power BI': (61, 80), 'Statistics': (81, 100)
}
OPTION_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

SET_A_KEYS = [None] * 101
SET_A_KEYS[1:21] = ['a','c','c','c','c','a','c','c','b','c','a','a','d','a','b','a b c d','c','d','a','b']
SET_A_KEYS[21:41] = ['a','d','b','a','c','b','a','b','d','c','c','a','b','c','a','b','d','b','a','b']
SET_A_KEYS[41:61] = ['c','c','c','b','b','a','c','b','d','a','c','b','c','c','a','b','b','a','a b','b']
SET_A_KEYS[61:81] = ['b','c','a','b','c','b','b','c','c','b','b','b','d','b','a','b','b','b','b','b']
SET_A_KEYS[81:101] = ['a','b','c','b','c','b','b','b','a','b','c','b','c','b','b','b','c','a','b','c']

SET_B_KEYS = [None] * 101
SET_B_KEYS[1:21] = ['a','b','d','b','b','d','c','c','a','c','a','b','d','c','c','a','c','b','d','c']
SET_B_KEYS[21:41] = ['a','a','b','a','b','a','b','b','c','c','b','c','b','c','a','a','a','b','b','a']
SET_B_KEYS[41:61] = ['b','a','d','b','c','b','b','b','b','b','c','a','c','a','c','c','b','a','b','c']
SET_B_KEYS[61:81] = ['b','b','b','d','c','b','b','a','b','b','b','c','a','d','b','b','d','a','b','a']
SET_B_KEYS[81:101] = ['b','c','b','a','c','b','b','b','b','d','c','d','b','b','b','c','c','b','b','c']

KEYS = {'A': SET_A_KEYS, 'B': SET_B_KEYS}