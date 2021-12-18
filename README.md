# Convolution with Haskell

Blog text available also at https://programmingwithjp.wordpress.com/2021/12/18/convolution-with-haskell/

Haskell is a functional programming language [1], which I have learned during the last few years. I have a bit of experience
also from other functional language Clojure [2], and for some reason I have found Haskell to make a lot more "sense"; can't
really put my finger on it though why it feels better.

I have tinkered with mathematical operation convolution [3] a few times before. To continue this trend and to
improve my Haskell, I will implement convolution with Haskell. First we will see how the naive version works out,
and then we will cycle the convolution through the frequency domain using discrete Fourier transform (DFT) [4].

Note that throughout this post when we use the term "convolution", we mean (of course) discrete convolution.

## Naive implementation

Convolution can be defined as

    y_n = sum h_n-k x_k  (eq. 1)

When we introduce kernel h and input x, they are assumed to have zeroes in places where they are not defined. What we
can see from the definition is that the kernel actually gets reversed and shifted and is then used to calculate a dot
product with the input elements. Other noteworthy thing is that that if we mark the number of non-zero elements in
vector a with |a|, we can deduce that
    
    |y| = |h| + |x| - 1  (eq. 2)

Armed with these observations, we can write our direct implementation for calculating n:th element of the convolution:

    convOne :: (Num a) => [a] -> [a] -> Int -> a
    convOne h x n    = sum (zipWith (*) kernel input)
        where kernel = drop (-begin) (reverse h)
              input  = drop begin x
              begin  = n - length h + 1

So there are few things here that probably need a bit of explanation. The first thing is the function *zipWith*. It
applies the given function (multiplication in this case) to the given lists; what is important for us, it works also
when the given lists have different lengths. The second thing is how we implement the "sliding window" of the
convolution. For the first values of n, only the last elements of the reversed kernel are used; after the whole kernel
is in use, we should start to shift the input. Again the the Haskell functions work in our favour. Function *drop* is
fine with receiving negative number (it does not drop anything), or number greater than the list length (it results
empty list).

Now when we have the way to calculate one element, we can write our full convolution function using list comprehension.

    conv :: (Num a) => [a] -> [a] -> [a]
    conv h x = [convOne h x n | n <- [0..(length x + length h - 2)]]

Let's check our convolution results against *conv* function in Octave:

    *Main> conv [1..5] [1..10]
    [1,4,10,20,35,50,65,80,95,110,114,106,85,50]

    octave:1> conv([1:5], [1:10])
    ans =

         1     4    10    20    35    50    65    80    95   110   114   106    85    50

Hooray!

## Convolution using discrete Fourier transform

Implementing convolution using Fourier transform is based on the fact that convolution in time domain equals
element-wise multiplication in frequency domain (and vice-versa). It is easy to see that the element-wise multiplication
is a lot easier calculation to do than the convolution. And since we have efficient method for the transform, Fast Fourier
Transform (FFT) [5], this is often a good solution for calculating the convolution.

### Fourier transform

We will use Cooley-Tukey FFT algorithm [6], which is the most common FFT (and what is usually meant when speaking about
FFT). The main idea there is to divide the calculation and recursively solve the subproblems. Haskell excels with
recursive functions, so this sounds like a good match. Before proceeding, there are few issues to keep in mind: DFT
assumes that the data is cyclic, and FFT works only for the lengths that are power of two. Luckily, zero-padding solves
both issues.

    import Data.Complex

    isPowerOfTwo :: Int -> Bool
    isPowerOfTwo 1 = True
    isPowerOfTwo x
        | mod x 2 == 0 = isPowerOfTwo (div x 2)
        | otherwise = False

    fftRaw :: (RealFloat a) => [a] -> Int -> Int -> [Complex a]
    fftRaw _ 0 _ = []
    fftRaw [] _  _ = []
    fftRaw (x0:_) 1 _ = [x0 :+ 0]
    fftRaw x n s = zipWith (+) x1 x2 ++ (zipWith (-) x1 x2)
        where x1 = fftRaw x (div n 2) (2 * s)
              x2 = zipWith (*) [exp (0 :+ (-2 * pi * fromIntegral k / (fromIntegral n))) | k <- [0..((div n 2) - 1)]] (fftRaw (drop s x) (div n 2) (2 * s))

    fft :: (RealFloat a) => [a] -> [Complex a]
    fft x
        | isPowerOfTwo n = fftRaw x n 1
        | otherwise = error "FFT works only for powers of two"
        where n = length x

Function *fft* is a convenience wrapper, while the *fftRaw* has the actual implementation. The implementation follows
the pseudocode from [7]. In Haskell complex numbers can be represented with format *re +: im* and require the importing
of *Data.Complex*. The inverse transform is almost the same; the complex phasor is just rotated in different direction,
and the final result has to be normalized with the element count.

    ifftRaw :: (RealFloat a) => [Complex a] -> Int -> Int -> [Complex a]
    ifftRaw _ 0 _ = []
    ifftRaw [] _  _ = []
    ifftRaw (x0:_) 1 _ = [x0]
    ifftRaw x n s = zipWith (+) x1 x2 ++ (zipWith (-) x1 x2)
        where x1 = ifftRaw x (div n 2) (2 * s)
              x2 = zipWith (*) [exp (0 :+ (2 * pi * fromIntegral k / (fromIntegral n))) | k <- [0..((div n 2) - 1)]] (ifftRaw (drop s x) (div n 2) (2 * s))

    ifft :: (RealFloat a) => [Complex a] -> [a]
    ifft x
        | isPowerOfTwo n = [realPart v / (fromIntegral n) | v <- ifftRaw x n 1]
        | otherwise = error "IFFT works only for powers of two"
        where n = length x

### FFT-based convolution

The implementation is very simple when we have *fft* and *ifft* implemented.

    convFft :: (RealFloat a) => [a] -> [a] -> [a]
    convFft h x
        | length x == length h = ifft (zipWith (*) (fft h) (fft x))
        | otherwise = error "Kernel and input data must have the same length"


As mentioned before, DFT assumes cyclic data, which causes the convolution to be also cyclic. In practice this would
mean that the output values in the beginning and in the end might get different values; however, when zero-padding is
used, the differences won't show up. We will use *replicate* to fill up the lists with zeros up to length which is power of two.

    *Main> h = concat [[1..5], replicate 11 0]
    *Main> x = concat [[1..10], replicate 6 0]
    *Main> convFft h x
    [1.0,4.000000000000007,10.0,20.0,35.0,49.99999999999999,65.0,80.0,95.0,110.0,114.0,106.0,85.0,50.00000000000001,7.105427357601002e-15,7.105427357601002e-15]
    *Main> 

So the values look first bit weird, but they are only suffering from the numerical inaccuracies, since the
implementation does now a decent amount of complicated floating point math. But when the values are compared to ones we
got with the naive implementation we can see they match.

Hooray again!

## References

[1] https://www.haskell.org/

[2] https://clojure.org/

[3] Steven W. Smith "The Scientist and Engineer's Guide to Digital Signal Processing" Chapter 6 http://www.dspguide.com/ch6/2.htm

[4] https://en.wikipedia.org/wiki/Discrete_Fourier_transform 

[5] https://en.wikipedia.org/wiki/Fast_Fourier_transform

[6] https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

[7] Steven. G. Johnson and Matteo Frigo "Implementing FFTs in practice" https://cnx.org/contents/ulXtQbN7@15/Implementing-FFTs-in-Practice


