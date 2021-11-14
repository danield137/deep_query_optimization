from __future__ import annotations

from dqo.db.execution_plan import ExecutionPlan

EXAMPLE_JSON_PLAN = [
  {
    "Plan": {
      "Node Type": "Aggregate",
      "Strategy": "Plain",
      "Partial Mode": "Simple",
      "Parallel Aware": False,
      "Startup Cost": 156719.16,
      "Total Cost": 156719.17,
      "Plan Rows": 1,
      "Plan Width": 4,
      "Actual Startup Time": 401.205,
      "Actual Total Time": 401.214,
      "Actual Rows": 1,
      "Actual Loops": 1,
      "Plans": [
        {
          "Node Type": "Nested Loop",
          "Parent Relationship": "Outer",
          "Parallel Aware": False,
          "Join Type": "Inner",
          "Startup Cost": 1000.42,
          "Total Cost": 156719.16,
          "Plan Rows": 1,
          "Plan Width": 4,
          "Actual Startup Time": 401.174,
          "Actual Total Time": 401.180,
          "Actual Rows": 0,
          "Actual Loops": 1,
          "Inner Unique": True,
          "Plans": [
            {
              "Node Type": "Gather",
              "Parent Relationship": "Outer",
              "Parallel Aware": False,
              "Startup Cost": 1000.00,
              "Total Cost": 156710.72,
              "Plan Rows": 1,
              "Plan Width": 4,
              "Actual Startup Time": 401.165,
              "Actual Total Time": 404.114,
              "Actual Rows": 0,
              "Actual Loops": 1,
              "Workers Planned": 2,
              "Workers Launched": 2,
              "Single Copy": False,
              "Plans": [
                {
                  "Node Type": "Seq Scan",
                  "Parent Relationship": "Outer",
                  "Parallel Aware": True,
                  "Relation Name": "dss_lineitem",
                  "Alias": "dss_lineitem",
                  "Startup Cost": 0.00,
                  "Total Cost": 155710.62,
                  "Plan Rows": 1,
                  "Plan Width": 4,
                  "Actual Startup Time": 365.735,
                  "Actual Total Time": 365.741,
                  "Actual Rows": 0,
                  "Actual Loops": 3,
                  "Filter": "((l_comment)::text ~~ '5'::text)",
                  "Rows Removed by Filter": 1999464
                }
              ]
            },
            {
              "Node Type": "Index Scan",
              "Parent Relationship": "Inner",
              "Parallel Aware": False,
              "Scan Direction": "Forward",
              "Index Name": "dss_part_pkey",
              "Relation Name": "dss_part",
              "Alias": "dss_part",
              "Startup Cost": 0.42,
              "Total Cost": 8.44,
              "Plan Rows": 1,
              "Plan Width": 8,
              "Actual Startup Time": 0.000,
              "Actual Total Time": 0.000,
              "Actual Rows": 0,
              "Actual Loops": 0,
              "Index Cond": "(p_partkey = dss_lineitem.l_partkey)",
              "Rows Removed by Index Recheck": 0
            }
          ]
        }
      ]
    },
    "Planning Time": 0.166,
    "Triggers": [
    ],
    "JIT": {
      "Worker Number": -1,
      "Functions": 19,
      "Options": {
        "Inlining": False,
        "Optimization": False,
        "Expressions": True,
        "Deforming": True
      },
      "Timing": {
        "Generation": 2.230,
        "Inlining": 0.000,
        "Optimization": 1.266,
        "Emission": 16.635,
        "Total": 20.131
      }
    },
    "Execution Time": 405.404
  }
]


def test_serialize_column():
    tree = ExecutionPlan.parse_analyze_result(EXAMPLE_PLAN)
    print(tree.pretty())
