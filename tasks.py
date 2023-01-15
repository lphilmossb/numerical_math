from invoke import task
from pathlib import Path
from pylint.lint import Run
from pylint.reporters.text import TextReporter

PROJECT_ROOT = Path('.')
PROJECT_DIRS = ('Computational_Mathematics_1')
EXCLUDE = ('tasks')
PASSING_SCORE = 10.0


@task
def lint(c):
    print('linting all .py files')
    for file in PROJECT_ROOT.glob('**/*.py'):
        fname = str(file.with_suffix(''))
        if fname not in EXCLUDE:
            rfile = file.with_suffix('.report.txt')
            score = 0.0
            with open(rfile, 'w', encoding='utf8') as rf:
                reporter = TextReporter(rf)
                result = Run([str(file)], do_exit=False, reporter=reporter)
                score = result.linter.stats.global_note
            print(f'{file} : {score}')
            if score == PASSING_SCORE:
                rfile.unlink()
