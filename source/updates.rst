Updates
=======

1.4.0
-----
* New key-word argument :py:data:`normalize` can be :py:data:`'short'` or :py:data:`'full'`. :py:data:`'short'` computes a simplified correlation coefficient that assumes the signal in every sliding window has a mean of zero (initial and default implementation of FMF). :py:data:`'full'` computes the full correlation coefficient and is slower. NB: :py:data:`'full'` cannot be used with :py:data:`arch='cpu'`.


1.3.0
-----
* :py:data:`arch` can now be :py:data:`'precise'` in addition to :py:data:`'cpu'` or :py:data:`'gpu'`. :py:data:`'precise'` is a CPU implementation that does not use an optimized summation algorithm to speed up the calculation of the sum of the squared data. Thus, :py:data:`'precise'` is less fast than :py:data:`'cpu'` but does not lose in accuracy when large amplitudes are encountered in the data (which can sometimes happen with :py:data:`'cpu'`).
* The sum of the squared templates is computed only once at the beginning.
* The station and component axes of the input arrays can be merged into a single axis of traces.

1.2.0
-----
* Fixed a bug in the computation of the sum of the squared data that occurred when some of the weights were equal to zero.

