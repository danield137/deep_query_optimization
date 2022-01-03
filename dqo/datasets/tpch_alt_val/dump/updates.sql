select count(distinct c_acctbal), count(c_acctbal) from customer;




UPDATE lineitem set l_quantity = 0.1 where lineitem.l_orderkey % 3 = 0;
UPDATE lineitem set l_discount = 0.5 where lineitem.l_orderkey % 3 = 1;
UPDATE lineitem set l_discount = 0.75 where lineitem.l_orderkey % 3 = 2;

UPDATE lineitem set l_quantity =  lineitem.l_orderkey % 100 where true;

UPDATE lineitem set l_receiptdate = '2000-02-01' where lineitem.l_orderkey % 10 = 1;
UPDATE lineitem set l_receiptdate = '2000-03-01' where lineitem.l_orderkey % 10 = 2;
UPDATE lineitem set l_receiptdate = '2000-04-01' where lineitem.l_orderkey % 10 = 3;
UPDATE lineitem set l_receiptdate = '2000-05-01' where lineitem.l_orderkey % 10 = 4;
UPDATE lineitem set l_receiptdate = '2000-06-01' where lineitem.l_orderkey % 10 = 5;
UPDATE lineitem set l_receiptdate = '2000-07-01' where lineitem.l_orderkey % 10 = 6;
UPDATE lineitem set l_receiptdate = '2000-08-01' where lineitem.l_orderkey % 10 = 7;
UPDATE lineitem set l_receiptdate = '2000-09-01' where lineitem.l_orderkey % 10 = 8;
UPDATE lineitem set l_receiptdate = '2000-10-01' where lineitem.l_orderkey % 10 = 9;

UPDATE lineitem set l_commitdate = '2000-02-01' where lineitem.l_orderkey % 10 = 1;
UPDATE lineitem set l_commitdate = '2000-03-01' where lineitem.l_orderkey % 10 = 2;
UPDATE lineitem set l_commitdate = '2000-04-01' where lineitem.l_orderkey % 10 = 3;
UPDATE lineitem set l_commitdate = '2000-05-01' where lineitem.l_orderkey % 10 = 4;
UPDATE lineitem set l_commitdate = '2000-06-01' where lineitem.l_orderkey % 10 = 5;
UPDATE lineitem set l_commitdate = '2000-07-01' where lineitem.l_orderkey % 10 = 6;
UPDATE lineitem set l_commitdate = '2000-08-01' where lineitem.l_orderkey % 10 = 7;
UPDATE lineitem set l_commitdate = '2000-09-01' where lineitem.l_orderkey % 10 = 8;
UPDATE lineitem set l_commitdate = '2000-10-01' where lineitem.l_orderkey % 10 = 9;

UPDATE lineitem set l_comment = 'no' where True;
UPDATE lineitem set l_comment = 'yes' where lineitem.l_orderkey % 10 = 9;


UPDATE orders set o_shippriority = o_orderkey % 3 where True;
UPDATE orders set o_custkey = o_orderkey % 100 where True;
UPDATE orders set o_totalprice = o_orderkey % 50 where True ;
UPDATE orders set o_comment = 'I want ' || (o_orderkey % 3)::text || 'start' where True;

UPDATE part set p_size = p_size % 20;
UPDATE part set p_retailprice = round(p_retailprice) * 1.0;
UPDATE part set p_comment = 'I want ' || (p_partkey % 3)::text || 'start' where True;


select 'a'  || 1::text || 'b';

UPDATE partsupp set ps_comment = 'I want ' || (ps_partkey % 25)::text || 'start' where true;
UPDATE partsupp set ps_supplycost = sqrt(ps_supplycost) / 2;

update supplier set s_acctbal = round(s_acctbal) % 1000 * 1000 where true;

update supplier set s_phone = '111-111-1111' where s_suppkey % 4 = 1;
update supplier set s_phone = '222-222-2222' where s_suppkey % 4 = 2;
UPDATE supplier set s_comment = 'I want ' || s_name  where True;

UPDATE region set r_comment = 'a' where true;