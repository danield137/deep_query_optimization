from dqo.relational.sql.ast import lex, yacc


def print_tokens():
    while True:
        tok = lex.token()
        if not tok:
            return
        print(tok)


def test_lexer_simple():
    lex.input('SELECT * FROM t WHERE a >= 1 AND b = 2')
    print_tokens()


def test_lexer_simple_case_insensetive():
    lex.input('select * from t where a > 1 and b = 2')
    print_tokens()


def test_lexer_and_or():
    lex.input('select * from t as d, t2 as d2 WHERE (d.a > 1 OR d.b = 2) and d.id = d2.id')
    print_tokens()


def test_parser_with_reserved_words():
    query = 'Select ind.id from indoor_and_outdoor_pools as ind where ind.price >= 100'
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_with_reserved_words2():
    query = 'Select ass.id from select_only as ass, not_the_best as ord where ass.price >= 100'
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_simple_reverse_condition():
    t = yacc.parse('Select e.id from employees as e where 100 > e.salary')
    t.pprint()


def test_parser_advance1():
    lex.input('select * from t WHERE ( t.id > 1 AND t.moshe = "moshe" )')
    print_tokens()
    t = yacc.parse('select * from t WHERE ( t.id > 1 AND t.moshe = "moshe" )')
    t.pprint()


def test_parser_advance_like():
    query = 'select * from t WHERE  t.id > 1 AND t.moshe = "moshe" AND t.david LIKE "a[]{}%" '
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advance_like_number():
    query = 'select * from t WHERE  t.id > 1 AND t.moshe = "moshe" AND t.david LIKE "[]{}1%" '
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advance_name_with_whitespace():
    query = 'select * from t WHERE  t.id > 1 AND t.moshe = "moshe moshe"'
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advance_name_with_special_chars():
    query = "select * from t WHERE  t.id > 1 AND t.moshe = '[mo]' "
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advance_like_numbers():
    query = "select * from t WHERE  t.id != 1 AND t.moshe LIKE '%1%'"
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advanced2():
    query = """
        SELECT MIN(cn.name) AS producing_company,        
                    MIN(miidx.info) AS rating,        
                    MIN(t.title) AS movie_about_winning 
                  FROM company_name AS cn,      
                    company_type AS ct,      
                    info_type AS it,      
                    info_type AS it2,      
                    kind_type AS kt,      
                    movie_companies AS mc,      
                    movie_info AS mi,      
                    movie_info_idx AS miidx,      
                    title AS t 
                  WHERE cn.country_code ='us'   
                    AND ct.kind ='production companies'   
                    AND it.info ='rating'   
                    AND it2.info ='release dates'   
                    AND kt.kind ='movie'   
                    AND t.title != ''   
                    AND (t.title LIKE 'Champion%' OR t.title LIKE 'Loser%') 
                    AND mi.movie_id = t.id 
                    AND it2.id = mi.info_type_id   
                    AND kt.id = t.kind_id   
                    AND mc.movie_id = t.id 
                    AND cn.id = mc.company_id 
                    AND ct.id = mc.company_type_id 
                    AND miidx.movie_id = t.id 
                    AND it.id = miidx.info_type_id 
                    AND mi.movie_id = miidx.movie_id 
                    AND mi.movie_id = mc.movie_id 
                    AND miidx.movie_id = mc.movie_id
            """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advanced_is_not_null():
    query = """
            SELECT MIN(n.name) AS cast_member_name,        
                MIN(pi.info) AS cast_member_info 
            FROM aka_name AS an,
                 cast_info AS pi,
                 info_type AS it
            WHERE an.name IS NULL AND 
                 it.info ='mini biography' AND 
                 pi.note IS NOT NULL """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_in_spaces():
    query = """
    SELECT * FROM tbl as t where t.name IN ('references', 'referenced in', 'features', 'featured in')
    """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parse_true_false():
    query = """Select e.id as eid
                From employees as e 
                where e.ok IS NOT FALSE AND e.name NOT LIKE '%DEAD%' AND e.alive IS TRUE """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parse_eq_true():
    query = """
            Select e.id as eid, d.id as did
            From employees as e ,
                 departments as d
            where (e.salary > 100 OR d.tech = TRUE) 
               AND d.id = e.dep_id
        """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_text():
    query = """SELECT MIN(mi.info) AS movie_budget,
                        MIN(mi_idx.info) AS movie_votes,
                        MIN(n.name) AS writer,
                        MIN(t.title) AS complete_violent_movie 
                FROM complete_cast AS cc,      
                    comp_cast_type AS cct1,      
                    comp_cast_type AS cct2,      
                    cast_info AS ci,      
                    info_type AS it1,      
                    info_type AS it2,      
                    keyword AS k,      
                    movie_info AS mi,      
                    movie_info_idx AS mi_idx,      
                    movie_keyword AS mk,      
                    name AS n,      
                    title AS t 
                WHERE cct1.kind = 'cast'   AND 
                      cct2.kind ='complete+verified'   AND 
                      ci.note IN ('(writer)','(head writer)','(written by)','(story)',   '(story editor)')   AND 
                      it1.info = 'genres'   AND 
                      it2.info = 'votes'   AND 
                      k.keyword IN ('murder',     'violence',     'blood',     'gore',     'death',     'female-nudity',     'hospital')   AND 
                      mi.info IN ('Horror',   'Action',   'Sci-Fi',   'Thriller',   'Crime',   'War')   AND 
                      n.gender = 'm'   AND 
                      t.id = mi.movie_id   AND 
                      t.id = mi_idx.movie_id   AND 
                      t.id = ci.movie_id   AND 
                      t.id = mk.movie_id   AND 
                      t.id = cc.movie_id   AND 
                      ci.movie_id = mi.movie_id   AND 
                      ci.movie_id = mi_idx.movie_id   AND 
                      ci.movie_id = mk.movie_id   AND 
                      ci.movie_id = cc.movie_id   AND 
                      mi.movie_id = mi_idx.movie_id   AND 
                      mi.movie_id = mk.movie_id   AND 
                      mi.movie_id = cc.movie_id   AND 
                      mi_idx.movie_id = mk.movie_id   AND 
                      mi_idx.movie_id = cc.movie_id   AND 
                      mk.movie_id = cc.movie_id   AND 
                      n.id = ci.person_id   AND 
                      it1.id = mi.info_type_id   AND 
                      it2.id = mi_idx.info_type_id   AND 
                      k.id = mk.keyword_id   AND 
                      cct1.id = cc.subject_id   AND 
                      cct2.id = cc.status_id"""
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_parenthesis():
    query = """SELECT MIN(mc.note) AS production_note
                FROM movie_companies AS mc
                WHERE  ( mc.note LIKE '%(co-production)%' )  """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_between():
    query = """
            SELECT *
            FROM tbl AS t
            WHERE  
                 t.production_year BETWEEN 1980 AND 2010"""
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_math():
    query = """
            SELECT *
            FROM tbl AS t
            WHERE t.production_year > (1.5 + 198.5) / (11 - 1) * 2"""
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_sample():
    query = """SELECT MIN(lt.link) AS link_type,        
                    MIN(t1.title) AS first_movie,        
                    MIN(t2.title) AS second_movie 
                FROM keyword AS k,      
                    link_type AS lt,      
                    movie_keyword AS mk,      
                    movie_link AS ml,      
                    title AS t1,      
                    title AS t2 
                WHERE k.keyword ='10,000-mile-club'   AND 
                    mk.keyword_id = k.id   AND 
                    t1.id = mk.movie_id   AND 
                    ml.movie_id = t1.id   AND 
                    ml.linked_movie_id = t2.id AND 
                    lt.id = ml.link_type_id   AND mk.movie_id = t1.id"""
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_advance_nested_and():
    t = yacc.parse('select * from t as tt WHERE ( tt.id > 1 or tt.moshe = "moshe" ) AND tt.id < 5 AND t.david LIKE "%1%"')
    t.pprint()


def test_parser_advance_or_and():
    t = yacc.parse('select * from t as d, tt as dd WHERE ( ( d.a > 1 OR d.b = 2 ) and d.id = dd.id )')
    t.pprint()


def test_parser_advance_name_with_number():
    t = yacc.parse('select * from t as d, t2 as d2 ')
    t.pprint()


def test_parser_advance_alias_from():
    query = """Select e.id as eid, d.id as did  From employees as e , departments as d where e.salary > 100 AND d.id = e.dep_id"""
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_many_select_format():
    q = 'SELECT * FROM t2 as b GROUP BY b.age HAVING b.age > 10 '
    lex.input(q)
    print_tokens()
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A GROUP BY b.age LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A ORDER BY a.id, b.id LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A GROUP BY b.age HAVING b.age > 10 '
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A GROUP BY b.age'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A ORDER BY a.id, b.id'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A '
    t = yacc.parse(q)
    t.pprint()


def test_parser_many2():
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" AND b.name LIKE "%D%" AND b.age in (1,2,3) GROUP BY b.age HAVING b.age > 10  ORDER BY a.id, b.id LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT a.A FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age HAVING b.age > 10  ORDER BY a.id, b.id'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT a, b, c FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age ORDER BY a.id, b.id LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT A.A, b.b FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age ORDER BY a.id, b.id'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age HAVING b.age > 10  LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age HAVING b.age > 10 '
    t = yacc.parse(q)
    t.pprint()


def test_parser_many3():
    query = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe_cohen = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age LIMIT 1 '
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()


def test_parser_many4():
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) ORDER BY a.id, b.id LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) ORDER BY a.id, b.id'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) GROUP BY b.age'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3) LIMIT 1'
    t = yacc.parse(q)
    t.pprint()
    q = 'SELECT * FROM t1 as a, t2 as b, A WHERE a.id = b.id AND A.moshe = "A" and b.name LIKE "%D%" and b.age in (1,2,3)'
    t = yacc.parse(q)
    t.pprint()


def test_nested():
    q = """
    SELECT a.id 
        FROM (SELECT id 
               from A 
               where id > 10 
               LIMIT 100
             ) as a, B as b 
        WHERE a.id < 100 and a.id = b.id 
        LIMIT 1
    """
    t = yacc.parse(q)
    t.pprint()


def test_nested2():
    query = """
    Select e.id as eid, d.id as did, managers.review
        From employees as e,
             departments as d,
            (select id from employees where job="manager") as managers
        where e.salary > 100 AND 
            d.id = e.dep_id AND
            managers.id = e.manager_id AND 
            managers.review = 1 
    """
    lex.input(query)
    print_tokens()
    t = yacc.parse(query)
    t.pprint()
