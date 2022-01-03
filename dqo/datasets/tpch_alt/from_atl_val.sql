select count(*) from customer; -- 150,000
select count(*) from lineitem; -- 4,423,658
select count(*) from nation; -- 25
select count(*) from orders; -- 1,500,000
select count(*) from part; -- 200,000
select count(*) from partsupp; -- 800,000
select count(*) from region; -- 5
select count(*) from supplier; -- 10,000

select * from lineitem limit 1;

create table lineitem_to_keep
as
select *
from lineitem
tablesample system (10);

select count(*) from lineitem_to_keep;

truncate table lineitem;
insert into lineitem
select *
from lineitem_to_keep;

drop table lineitem_to_keep;

select * from region;

do
$$
declare
  i record;
begin
  for i in 1..13 loop
    insert into region (r_regionkey,r_name, r_comment)
        Select region.r_regionkey + (select count(*) from region), chr(ascii('A') + (random() * 25)::integer) || region.r_name, substr(md5(random()::text), 0, 25) from region;

  end loop;
end;
$$
;

select count(*) from nation;

create sequence nation_key_seq
    start 25
    increment 1
    NO MAXVALUE
    CACHE 1;

insert into nation (n_nationkey, n_name, n_regionkey, n_comment)
        Select nextval('nation_key_seq'), 'Nation of ' || left(region.r_name, 15), region.r_regionkey, substr(md5(random()::text), 0, 25) from region;

insert into nation (n_nationkey, n_name, n_regionkey, n_comment)
        Select nextval('nation_key_seq'), 'Republic of ' || left(region.r_name, 10), region.r_regionkey, substr(md5(random()::text), 0, 25) from region;

insert into nation (n_nationkey, n_name, n_regionkey, n_comment)
        Select nextval('nation_key_seq'), 'State of ' || left(region.r_name, 15), region.r_regionkey, substr(md5(random()::text), 0, 25) from region;

select * from orders limit 10;

create table orders_to_keep
as
select *
from orders
tablesample system (50);

select count(*) from orders_to_keep;

truncate table orders;
insert into orders
select *
from orders_to_keep;

drop table orders_to_keep;



select * from customer limit 1;

update customer set c_nationkey=(random() * (select count(*) from nation)::integer) where true;

select * from supplier limit 1;

select max(s_suppkey)
from supplier;

create sequence supplier_key_seq
    start 10001
    increment 1
    NO MAXVALUE
    CACHE 1;

do
$$
declare
  i record;
begin
  for i in 1..3 loop
    insert into supplier (s_suppkey,s_nationkey, s_comment,s_name, s_address, s_phone, s_acctbal)
        Select nextval('supplier_key_seq'),
               (random() * (select count(*) from nation)::integer),
               substr(md5(random()::text), 0, 25),
               'Supplier#',
               substr(md5(random()::text), 0, 25),
               s_phone,
               s_acctbal
        from supplier;
  end loop;
end;
$$
;

update supplier set s_name = s_name || TO_CHAR(
        s_suppkey,
        '099999999'
    ) where s_name = 'Supplier#';

update supplier set s_name=REPLACE(s_name, ' ', '' ) where s_suppkey>=10000;

select *
from partsupp limit 1;

create table partsupp_to_keep
as
select *
from partsupp
tablesample system (50);

select count(*) from partsupp_to_keep;

truncate table partsupp;
insert into partsupp
select *
from partsupp_to_keep;

drop table partsupp_to_keep;

truncate table orders;

select count(*) from customer; -- 150,000
select count(*) from lineitem; -- 442,365
select count(*) from nation; -- 122,905
select count(*) from orders; -- 752,351
select count(*) from part; -- 200,000
select count(*) from partsupp; -- 399,435
select count(*) from region; -- 40,960
select count(*) from supplier; -- 80,000
