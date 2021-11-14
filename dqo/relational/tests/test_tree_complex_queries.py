from dqo.relational.sql.ast import AbstractSyntaxTree
from dqo.relational.tree import parser
from dqo.relational.tree.node import ProjectionNode


def test_parser_advance_lots_of_joins():
    query = """
    SELECT  MIN(cn.name) AS producing_company,
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
      WHERE cn.country_code ='[us]'   
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
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 11
    assert len(result.get_projections()) == 1
    assert len(result.get_projections()[0].columns) == 3
    assert len(result.get_selection_columns()) == 20
    assert len(result.relations) == 9
    assert isinstance(result.root, ProjectionNode)


def test_parser_advanced_in_clause_and_joins():
    query = """
SELECT MIN(n.name) AS cast_member_name,
    MIN(pi.info) AS cast_member_info 
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t 
WHERE an.name IS NOT NULL AND 
     (an.name LIKE '%a%' OR an.name LIKE 'A%') AND 
     it.info ='mini biography' AND 
     lt.link IN ('references','referenced in','features','featured in') AND 
     n.name_pcode_cf BETWEEN 'A' AND 'F' AND 
     (n.gender='m' OR (n.gender = 'f' AND n.name LIKE 'A%')) AND 
     pi.note IS NOT NULL   AND 
     t.production_year BETWEEN 1980 AND 2010   AND 
     n.id = an.person_id   AND 
     n.id = pi.person_id   AND 
     ci.person_id = n.id   AND 
     t.id = ci.movie_id   AND 
     ml.linked_movie_id = t.id   AND 
     lt.id = ml.link_type_id   AND 
     it.id = pi.info_type_id   AND 
     pi.person_id = an.person_id   AND 
     pi.person_id = ci.person_id   AND 
     an.person_id = ci.person_id   AND 
     ci.movie_id = ml.linked_movie_id"""

    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 11
    assert len(result.get_selection_columns()) == 19
    assert len(result.relations) == 8
    assert len(result.get_projections()) == 1


def test_sample():
    query = """
        SELECT MIN(mi.info) AS movie_budget, 
            MIN(mi_idx.info) AS movie_votes,  
            MIN(n.name) AS writer,  
            MIN(t.title) AS violent_liongate_movie 
        FROM cast_info AS ci,
            company_name AS cn,
            info_type AS it1,
            info_type AS it2,
            keyword AS k,
            movie_companies AS mc,
            movie_info AS mi,
            movie_info_idx AS mi_idx,
            movie_keyword AS mk,
            name AS n,
            title AS t 
        WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)')   AND 
            cn.name LIKE 'Lionsgate%' AND 
            it1.info = 'genres'   AND 
            it2.info = 'votes'   AND 
            k.keyword IN ('murder', 'violence','blood','gore','death','female-nudity','hospital') AND 
            mc.note LIKE '%(Blu-ray)%'   AND 
            mi.info IN ('Horror','Thriller')   AND 
            n.gender = 'm'   AND 
            t.production_year > 2000   AND 
            (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%')   AND 
            t.id = mi.movie_id   AND 
            t.id = mi_idx.movie_id   AND 
            t.id = ci.movie_id   AND 
            t.id = mk.movie_id   AND 
            t.id = mc.movie_id   AND 
            ci.movie_id = mi.movie_id   AND 
            ci.movie_id = mi_idx.movie_id   AND 
            ci.movie_id = mk.movie_id   AND 
            ci.movie_id = mc.movie_id   AND 
            mi.movie_id = mi_idx.movie_id   AND 
            mi.movie_id = mk.movie_id   AND 
            mi.movie_id = mc.movie_id   AND 
            mi_idx.movie_id = mk.movie_id   AND 
            mi_idx.movie_id = mc.movie_id   AND 
            mk.movie_id = mc.movie_id   AND 
            n.id = ci.person_id   AND 
            it1.id = mi.info_type_id   AND 
            it2.id = mi_idx.info_type_id   AND 
            k.id = mk.keyword_id   AND 
            cn.id = mc.company_id"""

    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 20
    assert len(result.get_selection_columns()) == 26
    assert len(result.relations) == 11
    assert len(result.get_projections()) == 1
    assert isinstance(result.root, ProjectionNode)


def test_multiple_joins():
    query = """
        SELECT MIN(mi.info) AS movie_budget, 
            MIN(n.name) AS writer,  
            MIN(t.title) AS violent_liongate_movie 
        FROM  
            movie_info AS mi,      
            name AS n,
            title AS t 
        WHERE 
            mi.info IN ('Horror','Thriller')   AND 
            n.gender = 'm'   AND 
            t.production_year > 2000   AND 
            (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%')   AND 
            t.id = mi.movie_id   AND 
            n.imdb_id = mi.movie_id"""

    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 2
    assert len(result.get_selection_columns()) == 7
    assert len(result.relations) == 3
    assert len(result.get_projections()) == 1
    assert isinstance(result.root, ProjectionNode)


def test_texty():
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
    result = parser.parse_sql(query)
    print(result.pretty())

    assert len(result.get_joins()) == 21
    assert len(result.get_selection_columns()) == 26
    assert len(result.relations) == 12
    assert len(result.get_projections()) == 1
    assert isinstance(result.root, ProjectionNode)


def test_hard_one():
    query = """
    SELECT MIN(kind_type.id),
    MIN(char_name.imdb_id),
    MIN(kind_type.kind),
    MIN(name.name_pcode_cf),
    MIN(char_name.name_pcode_nf)
FROM aka_name,
    aka_title,
    cast_info,
    char_name,
    keyword,
    kind_type,
    name,
    title
WHERE title.episode_nr = aka_name.person_id
    AND title.episode_of_id = aka_title.id
    AND title.imdb_id = cast_info.nr_order
    AND title.imdb_index = keyword.phonetic_code
    AND title.md5sum = kind_type.kind
    AND title.md5sum = name.name_pcode_cf
    AND title.season_nr = char_name.id
    AND title.title = name.imdb_index"""

    result = parser.parse_sql(query)
    print(result.pretty())
