import sys
import os
import csv
from pathlib import Path

# import the code under test
sys.path.insert(0, os.path.abspath(Path(os.path.dirname(__file__)) / '..'))
import detect_duplicate_img

def test_write_similarities_for_threshold():
    sim_tup_list = [
        ('a', 'b', 0.99),  # b is clone of a
        ('a', 'c', 0.8), ###
        ('a', 'd', 0.99),  # d is clone of a
        ('a', 'e', 0.3), ###
        ('a', 'f', 0.1), ###
        ('a', 'g', 0.90), ###
        ('b', 'a', 0.99),  # b is clone of a (already established)
        ('b', 'c', 0.99),  # c is clone of a (transitive -- c is clone of b, which is clone of a)
        ('b', 'dd', 0.5), ####
        ('z', 'a', 0.99),  # z is clone of a
        ('q', 'c', 0.99),  # q is clone of a (transitive -- q is clone of c, which is clone of b, which is clone of a)
        ('s', 't', 0.99),  # t is a clone of s
        ('q', 'x', 0.99),  # x is a clone of a (transitive q->c->b->a)
        ('v', 'q', 0.99),  # v is a clone if a (transitive q->c->b->a)
        ('v', 'l', 0.99),  # l is a clone of a (transitive v->q->c->b->a)
        ('cc', 'ee', 0.99), # ee is a clone of cc
        ('cc', 'boo', 0.5), # boo is NOT a clone of cc
        ('zoz', 'cc', 0.5), # zoz is NOT a clone of cc since 0.5
        ('ee', 'tt', 0.97), # tt is NOT a clone of cc since 0.97
        ('i', 'x', 0.99),  # i is a clone of a (transitive x->q->c->b->a)
        ('zz', 's', 0.99), # zz is a clone of s
        ('xx', 't', 0.99), # xx is a clone of s (transitive t->s)
        ('qq', 'a', 0.99)  # qq is a clone of a
    ]

    unique_csv = 'unique_test.csv'
    similar_csv = 'similar_test.csv'
    detect_duplicate_img.write_similarities_for_threshold(
        sim_tup_list,
        'unique_test.csv',
        'similar_test.csv',
        0.99
    )

    expected_unique = [
        'a',
        'e',
        'f',
        'g',
        'dd',
        's',
        'cc',
        'tt',
        'zoz',
        'boo'
    ]
    expected_similar = [
        'b',
        'd',
        'c',
        'z',
        'q',
        't',
        'x',
        'v',
        'l',
        'ee',
        'i',
        'zz',
        'xx',
        'qq'
    ]

    # confirm we detect orignals properly
    uniquefile_rows = []
    with open(unique_csv, 'r') as uniquefile:
        reader = csv.reader(uniquefile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            uniquefile_rows.append(row[0])
        assert sorted(expected_unique) == sorted(uniquefile_rows)

    # confirm we detect similar or 'clones' properly
    similarfile_rows = []
    with open(similar_csv, 'r') as similarfile:
        reader = csv.reader(similarfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            similarfile_rows.append(row[0])
        assert sorted(expected_similar) == sorted(similarfile_rows)

    # confirm the unique and similar sets are disjoint
    assert not any(i in uniquefile_rows for i in similarfile_rows)

    # cleanup
    os.remove(unique_csv)
    os.remove(similar_csv)
