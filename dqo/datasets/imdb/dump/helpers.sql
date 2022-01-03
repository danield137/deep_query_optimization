--
-- delete from publ) ic.name UNION ALL
-- delete from ch) ar_name UNION ALL
-- delete from compa) ny_name UNION ALL
-- delete from kind_type UNION ALL
-- delete from title UNION ALL
-- delete from company_type UNION ALL
-- delete from a) ka_name UNION ALL
-- delete from aka_title UNION ALL
-- delete from role_type UNION ALL
-- delete from cast_info UNION ALL
-- delete from comp_cast_type UNION ALL
-- delete from complete_cast UNION ALL
-- delete from info_type UNION ALL
-- delete from link_type UNION ALL
-- delete from keyword UNION ALL
-- delete from movie_keyword UNION ALL
-- delete from movie_link UNION ALL
-- delete from movie_info UNION ALL
-- delete from movie_companies UNION ALL
-- delete from person_info UNION ALL

select count(*) as cnt, max('name') as name
from name
UNION ALL
select count(*) as cnt, max('char_name') as name
from char_name
UNION ALL
select count(*) as cnt, max('company_name') as name
from company_name
UNION ALL
select count(*) as cnt, max('kind_type') as name
from kind_type
UNION ALL
select count(*) as cnt, max('title') as name
from title
UNION ALL
select count(*)            as cnt,
       max('company_type') as name
from company_type
UNION ALL
select count(*)        as cnt,
       max('aka_name') as name
from aka_name
UNION ALL
select count(*)         as cnt,
       max('aka_title') as name
from aka_title
UNION ALL
select count(*) as cnt, max('role_type') as name
from role_type
UNION ALL
select count(*) as cnt, max('cast_info') as name
from cast_info
UNION ALL
select count(*) as cnt, max('comp_cast_type') as name
from comp_cast_type
UNION ALL
select count(*) as cnt, max('complete_cast') as name
from complete_cast
UNION ALL
select count(*) as cnt, max('info_type') as name
from info_type
UNION ALL
select count(*) as cnt, max('link_type') as name
from link_type
UNION ALL
select count(*) as cnt, max('keyword') as name
from keyword
UNION ALL
select count(*) as cnt, max('movie_keyword') as name
from movie_keyword
UNION ALL
select count(*) as cnt, max('movie_link') as name
from movie_link
UNION ALL
select count(*) as cnt, max('movie_info') as name
from movie_info
UNION ALL
select count(*) as cnt, max('movie_companies') as name
from movie_companies
UNION ALL
select count(*) as cnt, max('person_info') as name
from person_info
order by cnt desc