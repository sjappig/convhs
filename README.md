# Convolution with Haskell

Haskell is a functional programming language [1], which I have played around with last few years. I have a bit of experience
also from other functional language Clojure [2], and for some reason I have found Haskell to make a lot more "sense"; can't
really put my finger on it though, might be just about personal preferences.

In this blog I have tinkered with mathematical operation convolution a few times before [3, 4]. To continue this trend
and to improve my Haskell, I will implement convolution with Haskell. First we will see how the naive version works out,
and then we will cycle the convolution through the frequency domain using discrete Fourier transform (DFT) [5].

Note that throughout this post when we use the term "convolution", we mean (of course) discrete convolution.

## Naive implementation

Convolution is defined as

    y_n = sum h_n-k x_k  (eq. 1)

When we introduce kernel h and input x, they are assumed to have zeroes in places where they are not defined. What we
can see from the definition is that the kernel actually gets reversed and shifted and is then used to calculate a dot
product with the input elements. Other noteworthy thing is that that if we mark the number of non-zero elements in
vector a with |a|, we can deduce that
    
    |y| = |h| + |x| - 1  (eq. 2)

Armed with these observations, we can write our direct implementation for calculating n:th element of the output:

    convOne :: (Num a) => [a] -> [a] -> Int -> a
    convOne h x n    = sum (zipWith (*) kernel input)
        where kernel = drop (-begin) (reverse h)
              input  = drop begin x
              begin  = n - length h + 1

So there are few things here that probably need a bit of explanation. The first thing is the function zipWith. It
applies the given function (multiplication in this case) to the given lists; what is important for us, it works also
when the given lists have different lengths. The second thing is how we implement the "sliding window" of the
convolution. For the first values of n, only the last elements of the reversed kernel are used; after the whole kernel
is in use, we should start to shift the input. Again the the Haskell functions work in our favour. Function drop is
fine with receiving negative number (it does not drop anything), or number greater than the list length (it results
empty list).

Now when we have the way to calculate one element, we can write our full convolution function that uses list
comprehension.

    conv :: (Num a) => [a] -> [a] -> [a]
    conv h x = [convOne h x n | n <- [0..(length x + length h - 2)]]

Let's check our convolution results using Octave:

    *Main> conv [1..5] [1..10]
    [1,4,10,20,35,50,65,80,95,110,114,106,85,50]

    octave:1> conv([1:5], [1:10])
    ans =

         1     4    10    20    35    50    65    80    95   110   114   106    85    50

Hooray!

# Convolution using discrete Fourier transform

Implementing convolution using Fourier transform is based on the fact that convolution in time domain equals
element-wise multiplication in frequency domain (and vice-versa). It is easy to see that the element-wise multiplication
is a lot easier calculation to do than the convolution. And since we have efficient methods for the transform and its
inverse, Fast Fourier Transform (FFT) and it's inverse (IFFT) [6], this is often a good solution for calculating the
convolution.

We will use Cooley-Tukey FFT algorithm [7], which is the most common FFT (and what is usually meant when speaking about
FFT). The main idea there is to divide the calculation and recursively solve the subproblems. Haskell excels with
recursive functions, so this sounds like a good match. Before proceeding, there are few things to keep in mind: DFT
assumes that the data is cyclic, and FFT works only for the lengths that are power of two. Luckily, zero-padding solves
both issues.

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

