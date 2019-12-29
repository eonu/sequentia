.. _fft:

Discrete Fourier Transform (``fft``)
====================================

The Discrete Fourier Transform (DFT) converts the observation sequence into a real-valued,
same-length sequence of equally-spaced samples of the
`discrete-time Fourier transform <https://en.wikipedia.org/wiki/Discrete-time_Fourier_transform>`_.

The popular `Fast Fourier Transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_ (FFT) implementation is used to efficiently compute the DFT.

For further information, please see the `preprocessing tutorial notebook <https://nbviewer.jupyter.org/github/eonu/sequentia/blob/master/notebooks/2%20-%20Preprocessing%20%28Tutorial%29.ipynb#Discrete-Fourier-Transform-(fft)>`_.

Example
-------

.. literalinclude:: ../../_includes/examples/preprocessing/fft.py
    :language: python
    :linenos:

API reference
-------------

.. automodule:: sequentia.preprocessing
   :noindex:
.. autofunction:: fft