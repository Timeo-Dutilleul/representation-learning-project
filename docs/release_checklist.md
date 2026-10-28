# Public release checklist

Run these checks from the repository root before pushing:

1. Review the working tree and staged changes:

   ```sh
   git status --short
   git diff --check
   git diff --cached --check
   ```

2. Review every path that would be committed, including untracked files:

   ```sh
   git ls-files
   git status --short --untracked-files=all
   git diff --cached --name-only
   ```

   Confirm that no `results/`, dataset/cache, checkpoint, raw array, dashboard,
   generated audit manifest, private note, local path, secret, or environment
   file is tracked or staged. Remember that `.gitignore` does not untrack a
   file already in Git; inspect the index before using `git add .`.

3. Review ignored files and candidate additions:

   ```sh
   git check-ignore -v results/experiment-example/dashboard.html data/example.pt local/notes.md
   git status --short --ignored
   ```

4. Run the test suite without launching an experiment:

   ```sh
   PYTHONPATH=.. python -m unittest discover -s tests -v
   ```

5. Validate the public experiment configuration without training:

   ```sh
   PYTHONPATH=.. python -m redundancy_project.experiment_catalog --help
   PYTHONPATH=.. python -m redundancy_project.launch_experiment --help
   ```

6. Review the report source, references, licence scope, README instructions,
   and this checklist. Compile the report and review the resulting PDF before
   adding it to the public release.

Do not run the experiment as part of release checks. The 32-run launch is a
separate, expensive action and should only be started deliberately after the
configuration and required local dataset are confirmed.
