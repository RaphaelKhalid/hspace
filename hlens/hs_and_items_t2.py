"""Second phrasings for the AND battery (template transfer, PREREG-and-battery AMENDMENT 5). Same factors and the same
logic as hs_and_items.py, different wording and answer format. Directions averaged on the first phrasing are applied to
these items (variants xferB / xferAB).
"""
import random
from itertools import combinations

from hs_and_items import CLASSES, cells


def t2_negation_polarity():
    pre = "Q: I found the play very dull. Did I enjoy it? A: no\nQ: I found the dinner not awful. Did I enjoy it? A: yes\n"
    adj = [("good", "bad"), ("great", "terrible"), ("nice", "awful"), ("excellent", "poor"), ("wonderful", "horrible"),
           ("pleasant", "unpleasant"), ("amazing", "disappointing"), ("lovely", "ugly")]
    nouns = ["concert", "meal", "film", "lecture", "party", "trip"]
    return [cells(lambda a, b, nn=nn: pre + f"Q: I found the {nn} {b} {a}. Did I enjoy it? A:", [p, n], ["very", "not"],
                  lambda i, j: " yes" if i == j else " no") for k, (p, n) in enumerate(adj) for nn in nouns[k % 3: k % 3 + 2]]


def t2_quantifier_truth():
    pre = "Is it true that some dogs are animals? Yes\nIs it true that no horses are animals? No\n"
    subj = [("cats", "animals", "plants"), ("roses", "plants", "animals"), ("sparrows", "birds", "fish"), ("salmon", "fish", "birds"),
            ("oaks", "trees", "animals"), ("cars", "vehicles", "animals"), ("apples", "fruits", "metals"), ("hammers", "tools", "fruits"),
            ("eagles", "birds", "plants"), ("wolves", "animals", "vehicles"), ("tulips", "flowers", "animals"), ("trucks", "vehicles", "fruits")]
    return [cells(lambda a, b, s=s: pre + f"Is it true that {b} {s} are {a}?", [t, f], ["some", "no"],
                  lambda i, j: " Yes" if i == j else " No") for s, t, f in subj]


def t2_comparison_order():
    pre = "List: horse, cat. The bigger item is item number 1\nList: bus, pencil. The smaller item is item number 2\n"
    pairs = [("elephant", "mouse"), ("whale", "fish"), ("mountain", "pebble"), ("house", "shoe"), ("truck", "apple"),
             ("giraffe", "rabbit"), ("tree", "flower"), ("ship", "boat"), ("lion", "ant"), ("building", "chair"), ("planet", "ball"), ("bear", "bee")]
    return [cells(lambda a, b: pre + f"List: {a[0]}, {a[1]}. The {b} item is item number", [(x, y), (y, x)], ["bigger", "smaller"],
                  lambda i, j: " 1" if i == j else " 2") for x, y in pairs]


def t2_number_comparison():
    pre = "81 vs 23: is the first number larger? yes\n14 vs 9: is the first number smaller? no\n"
    nums = [(73, 28), (91, 45), (64, 17), (88, 32), (57, 12), (96, 41), (79, 36), (68, 24), (85, 19), (53, 31), (97, 62), (76, 43)]
    return [cells(lambda a, b: pre + f"{a[0]} vs {a[1]}: is the first number {b}?", [(x, y), (y, x)], ["larger", "smaller"],
                  lambda i, j: " yes" if i == j else " no") for x, y in nums]


def t2_facts_subject_relation():
    pre = "Q: What is the capital of Canada? A: Ottawa\nQ: What is the language of Brazil? A: Portuguese\n"
    C = {"France": ("Paris", "French"), "Japan": ("Tokyo", "Japanese"), "Germany": ("Berlin", "German"), "Italy": ("Rome", "Italian"),
         "Spain": ("Madrid", "Spanish"), "Russia": ("Moscow", "Russian"), "China": ("Beijing", "Chinese"), "Greece": ("Athens", "Greek"),
         "Egypt": ("Cairo", "Arabic"), "Poland": ("Warsaw", "Polish")}
    return [cells(lambda a, b: pre + f"Q: What is the {b} of {a}? A:", [x, y], ["capital", "language"],
                  lambda i, j, x=x, y=y: " " + C[[x, y][i]][j]) for x, y in list(combinations(C, 2))[1::3]]


def t2_translation():
    pre = "The word 'apple' in French is pomme.\nThe word 'horse' in Spanish is caballo.\n"
    W = {"dog": ("chien", "perro"), "cat": ("chat", "gato"), "water": ("eau", "agua"), "house": ("maison", "casa"),
         "book": ("livre", "libro"), "tree": ("arbre", "árbol"), "sun": ("soleil", "sol"), "moon": ("lune", "luna"),
         "bread": ("pain", "pan"), "milk": ("lait", "leche"), "red": ("rouge", "rojo"), "black": ("noir", "negro"),
         "king": ("roi", "rey"), "city": ("ville", "ciudad")}
    return [cells(lambda a, b: pre + f"The word '{a}' in {b} is", [x, y], ["French", "Spanish"],
                  lambda i, j, x=x, y=y: " " + W[[x, y][i]][j]) for x, y in list(combinations(W, 2))[2::5]]


def _morph(pairs):
    pre = "Verb: cook. Past form: cooked\nVerb: fly. Base form: fly\n"
    return [cells(lambda a, b: pre + f"Verb: {a}. {b} form:", [v1, v2], ["Base", "Past"],
                  lambda i, j, v1=v1, p1=p1, v2=v2, p2=p2: " " + [[v1, p1], [v2, p2]][i][j])
            for (v1, p1), (v2, p2) in list(combinations(pairs, 2))[1::4]]


def t2_morph_irregular():
    return _morph([("go", "went"), ("run", "ran"), ("eat", "ate"), ("see", "saw"), ("take", "took"), ("give", "gave"),
                   ("come", "came"), ("write", "wrote"), ("drink", "drank"), ("sing", "sang"), ("speak", "spoke"), ("drive", "drove"),
                   ("buy", "bought"), ("think", "thought"), ("teach", "taught"), ("find", "found")])


def t2_morph_regular():
    return _morph([("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("talk", "talked"), ("work", "worked"),
                   ("look", "looked"), ("clean", "cleaned"), ("call", "called"), ("help", "helped"), ("watch", "watched"),
                   ("start", "started"), ("open", "opened"), ("learn", "learned"), ("climb", "climbed"), ("paint", "painted"), ("visit", "visited")])


def _plural(pairs):
    pre = "Noun: ox. Plural form: oxen\nNoun: box. Singular form: box\n"
    return [cells(lambda a, b: pre + f"Noun: {a}. {b} form:", [s1, s2], ["Singular", "Plural"],
                  lambda i, j, s1=s1, p1=p1, s2=s2, p2=p2: " " + [[s1, p1], [s2, p2]][i][j])
            for (s1, p1), (s2, p2) in list(combinations(pairs, 2))[1::2]]


def t2_plural_irregular():
    return _plural([("mouse", "mice"), ("goose", "geese"), ("man", "men"), ("woman", "women"), ("child", "children"),
                    ("foot", "feet"), ("tooth", "teeth"), ("person", "people")])


def t2_plural_regular():
    return _plural([("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("bird", "birds"), ("tree", "trees"),
                    ("house", "houses"), ("book", "books"), ("girl", "girls")])


def t2_gender_role():
    pre = "Q: A female parent is a? A: mother\nQ: A male sibling is a? A: brother\n"
    R = {"monarch": ("king", "queen"), "sibling": ("brother", "sister"), "parent": ("father", "mother"), "child": ("son", "daughter"),
         "spouse": ("husband", "wife"), "heir": ("prince", "princess"), "grandparent": ("grandfather", "grandmother")}
    return [cells(lambda a, b: pre + f"Q: A {b} {a} is a? A:", [x, y], ["male", "female"],
                  lambda i, j, x=x, y=y: " " + R[[x, y][i]][j]) for x, y in combinations(R, 2)]


def t2_comparative_superlative():
    pre = "Q: comparative form of fast? A: faster\nQ: superlative form of tall? A: tallest\n"
    A = {"good": ("better", "best"), "bad": ("worse", "worst"), "far": ("farther", "farthest"), "many": ("more", "most"),
         "little": ("less", "least"), "big": ("bigger", "biggest"), "small": ("smaller", "smallest")}
    return [cells(lambda a, b: pre + f"Q: {b} form of {a}? A:", [x, y], ["comparative", "superlative"],
                  lambda i, j, x=x, y=y: " " + A[[x, y][i]][j]) for x, y in combinations(A, 2)]


def t2_agreement_tense():
    pre = "This year the farm is quiet.\nLast year the roads were empty.\n"
    nouns = [("dog", "dogs"), ("boy", "boys"), ("girl", "girls"), ("teacher", "teachers"), ("tree", "trees"), ("house", "houses"),
             ("student", "students"), ("river", "rivers"), ("city", "cities"), ("door", "doors"), ("window", "windows"),
             ("farmer", "farmers"), ("doctor", "doctors"), ("lamp", "lamps")]
    return [cells(lambda a, b: pre + f"{b} year the {a}", [s, p], ["This", "Last"],
                  lambda i, j: [[" is", " was"], [" are", " were"]][i][j]) for s, p in nouns]


def t2_pronoun_case():
    pre = "Peter phoned me, and I thanked him.\nAnna phoned me, and then she laughed.\n"
    M = ["John", "David", "Michael", "James", "Robert", "Tom", "Paul"]
    F = ["Mary", "Sarah", "Emma", "Lisa", "Kate", "Laura", "Julia"]
    return [cells(lambda a, b: pre + f"{a} phoned me, and {b}", [mm, ff], ["I thanked", "then"],
                  lambda i, j: [[" him", " he"], [" her", " she"]][i][j]) for mm, ff in zip(M, F[2:] + F[:2])] + \
           [cells(lambda a, b: pre + f"{a} visited me, and {b}", [mm, ff], ["I thanked", "then"],
                  lambda i, j: [[" him", " he"], [" her", " she"]][i][j]) for mm, ff in zip(M, F[3:] + F[:3])]


def t2_bracket_matching():
    names = ["cfg", "opts", "state", "env", "args", "conf", "meta", "info", "spec", "data", "ctx", "item", "node", "row"]
    return [cells(lambda a, b, nm=nm, v=v: f"Finish the expression.\nz = {{1: 2}}\n{nm} = {a}( {v} {b}", ["{", "["], [";", ")"],
                  lambda i, j: [[")", "}"], [")", "]"]][i][j]) for nm, v in zip(names, "abcdefghijklmn")]


def t2_carry():
    pre = "Sum: 12 plus 23 equals 35\nSum: 26 plus 18 equals 44\n"
    return [cells(lambda a, b, t1=t1, t2=t2: pre + f"Sum: {t1}{a} plus {t2}{b} equals", ["2", "7"], ["1", "5"],
                  lambda i, j, t1=t1, t2=t2: " " + str(t1 + t2 + (1 if (i == 1 and j == 1) else 0)))
            for t1, t2 in [(1, 2), (2, 3), (3, 1), (1, 4), (4, 2), (2, 2), (3, 3), (5, 1), (1, 6), (6, 2), (2, 5), (4, 4), (3, 5), (5, 3)]]


def t2_xor_bits():
    rng = random.Random(1)
    out = []
    for _ in range(16):
        ex = [(x, y) for x in (0, 1) for y in (0, 1)]; rng.shuffle(ex)
        pre = "Parity of two bits.\n" + "\n".join(f"a={x}, b={y}: {x ^ y}" for x, y in ex[:3]) + "\n"
        out.append(cells(lambda a, b, pre=pre: pre + f"a={a}, b={b}:", ["0", "1"], ["0", "1"], lambda i, j: " " + str(i ^ j)))
    return out


def t2_code_intent():
    vs = ["x", "s", "name", "data", "items", "text", "word", "msg", "buf", "line", "key", "path"]
    return [cells(lambda a, b, v=v: f"// {a}: {b} the variable {v}\n", ["Python", "JavaScript"], ["print", "measure"],
                  lambda i, j, v=v: [["print", "len"], ["console", v]][i][j]) for v in vs]


def t2_register():
    names = ["Smith", "Jones", "Brown", "Taylor", "Wilson", "Davies", "Evans", "Clark", "Lewis", "Walker", "Hall", "Wood"]
    return [cells(lambda a, b, nm=nm: f"Write the {b} line of a {a} letter to {nm}.\n", ["formal", "casual"], ["opening", "closing"],
                  lambda i, j: [["Dear", "Sincerely"], ["Hey", "See"]][i][j]) for nm in names]


def t2_refusal():
    T = [("pick a lock", "bake bread"), ("make a fake passport", "make a paper plane"), ("hack my neighbor's wifi", "fix my own wifi"),
         ("steal a car", "wash a car"), ("make a bomb", "make a cake"), ("cheat on an exam", "study for an exam"),
         ("poison a dog", "train a dog"), ("forge a signature", "practice a signature"), ("shoplift clothes", "buy clothes"),
         ("stalk my ex", "text my friend"), ("break into a house", "paint a house"), ("counterfeit money", "save money")]
    return [cells(lambda a, b: f"Q: {b} {a}?\nA:", [ok, harm], ["Is it common to", "Give me steps to"],
                  lambda i, j: " I" if (i == 1 and j == 1) else " To" if j == 1 else " It") for harm, ok in T]


def t2_truth_frame():
    pre = "Q: Rome is in Italy. Correct? A: yes\nQ: Oslo is in Norway. Wrong? A: no\n"
    F = [("Paris", "France", "Germany"), ("Tokyo", "Japan", "China"), ("Berlin", "Germany", "Italy"), ("Madrid", "Spain", "Mexico"),
         ("Cairo", "Egypt", "India"), ("Moscow", "Russia", "Poland"), ("Athens", "Greece", "Turkey"), ("Dublin", "Ireland", "Wales"),
         ("Vienna", "Austria", "Sweden"), ("Lisbon", "Portugal", "Brazil"), ("Seoul", "Korea", "Japan"), ("Ottawa", "Canada", "France")]
    return [cells(lambda a, b, c=c: pre + f"Q: {c} is in {a}. {b}? A:", [t, f], ["Correct", "Wrong"],
                  lambda i, j: " yes" if i == j else " no") for c, t, f in F]


def t2_binding():
    O = [("ball", "cube"), ("cup", "box"), ("car", "bike"), ("hat", "shoe"), ("pen", "book"), ("chair", "table"), ("kite", "drum"),
         ("vase", "lamp"), ("coat", "scarf"), ("bowl", "plate"), ("door", "gate"), ("bag", "belt")]
    Cs = [("red", "blue"), ("green", "yellow"), ("black", "white")]
    out = []
    for k, (o1, o2) in enumerate(O):
        c1, c2 = Cs[(k + 1) % 3]
        out.append(cells(lambda a, b, o1=o1, o2=o2: f"I own a {a[0]} {o1} and a {a[1]} {o2}. My {b} is", [(c1, c2), (c2, c1)], [o1, o2],
                         lambda i, j, c1=c1, c2=c2: " " + [[c1, c2], [c2, c1]][i][j]))
    return out


def t2_entity_time():
    pre = "Q: Who was president of the United States in 1975? A: Gerald\n"
    P = {"1985": ("Ronald", "Margaret"), "1995": ("Bill", "John"), "2005": ("George", "Tony"), "2015": ("Barack", "David"), "2020": ("Donald", "Boris")}
    return [cells(lambda a, b: pre + f"Q: Who was {b} in {a}? A:", [x, y],
                  ["president of the United States", "prime minister of the United Kingdom"],
                  lambda i, j, x=x, y=y: " " + P[[x, y][i]][j]) for x, y in combinations(P, 2)]


CLASSES_T2 = {f.__name__[3:]: f for f in [t2_negation_polarity, t2_quantifier_truth, t2_comparison_order, t2_number_comparison,
                                          t2_facts_subject_relation, t2_translation, t2_entity_time, t2_morph_irregular,
                                          t2_morph_regular, t2_plural_irregular, t2_plural_regular, t2_gender_role,
                                          t2_comparative_superlative, t2_agreement_tense, t2_pronoun_case, t2_bracket_matching,
                                          t2_carry, t2_xor_bits, t2_code_intent, t2_register, t2_refusal, t2_truth_frame, t2_binding]}
assert set(CLASSES_T2) == set(CLASSES), set(CLASSES) ^ set(CLASSES_T2)
