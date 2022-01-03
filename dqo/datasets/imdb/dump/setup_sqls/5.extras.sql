create type _stats_agg_accum_type AS (
	n bigint,
	min double precision,
	max double precision,
	m1 double precision,
	m2 double precision,
	m3 double precision,
	m4 double precision
);

create type _stats_agg_result_type AS (
	count bigint,
	min double precision,
	max double precision,
	mean double precision,
	variance double precision,
	skewness double precision,
	kurtosis double precision
);

create or replace function _stats_agg_accumulator(_stats_agg_accum_type, double precision)
returns _stats_agg_accum_type AS '
DECLARE
	a ALIAS FOR $1;
	x alias for $2;
	n1 bigint;
	delta double precision;
	delta_n double precision;
	delta_n2 double precision;
	term1 double precision;
BEGIN
	n1 = a.n;
	a.n = a.n + 1;
	delta = x - a.m1;
	delta_n = delta / a.n;
	delta_n2 = delta_n * delta_n;
	term1 = delta * delta_n * n1;
	a.m1 = a.m1 + delta_n;
	a.m4 = a.m4 + term1 * delta_n2 * (a.n*a.n - 3*a.n + 3) + 6 * delta_n2 * a.m2 - 4 * delta_n * a.m3;
	a.m3 = a.m3 + term1 * delta_n * (a.n - 2) - 3 * delta_n * a.m2;
	a.m2 = a.m2 + term1;
	a.min = least(a.min, x);
	a.max = greatest(a.max, x);
	RETURN a;
END;
'
language plpgsql;

create or replace function _stats_agg_finalizer(_stats_agg_accum_type)
returns _stats_agg_result_type AS '
BEGIN
	RETURN row(
		$1.n,
		$1.min,
		$1.max,
		$1.m1,
		$1.m2 / nullif(($1.n - 1.0), 0),
		case when $1.m2 = 0 then null else sqrt($1.n) * $1.m3 / nullif(($1.m2 ^ 1.5), 0) end,
		case when $1.m2 = 0 then null else $1.n * $1.m4 / nullif(($1.m2 * $1.m2) - 3.0, 0) end
	);
END;
'
language plpgsql;

create aggregate stats_agg(double precision) (
	sfunc = _stats_agg_accumulator,
	stype = _stats_agg_accum_type,
	finalfunc = _stats_agg_finalizer,
	initcond = '(0,,, 0, 0, 0, 0)'
);