# DenserFlow

This was part of an assignment that I and @arjun-prakash wrote together. The assignment consisted of multiple steps: writing the library, creating and optimising a model (using this code) on given dataset, and writing a report justifying our design decisions and the model itself. The report and model have been left out of this repo, leaving just the basic library part. Please note @arjun-prakash did write a fair bit, just due to me wiping and re-creating the repository I have wiped the git history.

To setup:

1) ```virtualenv venv && source ./venv/bin/activate```
2) ```pre-commit install``` for pre-commit hooks
3) ```pip install -r requirements.txt``` for requirements
4) ```mkdir dataset``` and place the train and label data files inside (for whatever dataset you need).

See ```example.py``` for an basic way to use this library.

Note that this was made for an assignment, so its nowhere near production quality - go use a 'real' neural network library if you are trying to do anything remotely complex. Overflows are common with this code and there are things that could probably be tightened up. But hopefully its useful regardless!

I've also added a new demo program, ```mnist.py```, which shows some basic usage of the library. even a very simple neural net as implemented in DenserFlow can get over 90% accuracy on MNIST - sure, that's easy, but its still nice to see, especially since only dense layers are being used (with batchnorm and dropout).

The 'Documentation' folder is largely just auto-generated documentation, through the docstrings added in the library itself - so feel free to either look at the pages in the documentation folder, or just right at the code itself (which is also commented moderately well).
