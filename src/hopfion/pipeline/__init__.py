"""Manufacturing-style pipeline.

Recipes are YAML files. ``runner.run(recipe)`` orchestrates a single run with
metric callbacks + QC verdicts. ``batch`` sweeps parameters and seeds and
reports yield statistics. ``cli`` exposes the command-line surface.

A typical run::

    from hopfion.pipeline.recipe import RecipeConfig
    from hopfion.pipeline.runner import run
    result = run(RecipeConfig.from_yaml("recipes/single_hopfion.yaml"))
    print(result.qc.verdict)   # 'ACCEPT' or 'FAIL'
"""
