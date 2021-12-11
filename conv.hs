{-
 - *Main Data.Complex> h = concat [[1..5], replicate 11 0]
 - *Main Data.Complex> x = concat [[1..10], replicate 6 0]
 - *Main Data.Complex> convFft h x
 - [1.0,4.000000000000007,10.0,20.0,35.0,49.99999999999999,65.0,80.0,95.0,110.0,114.0,106.0,85.0,50.00000000000001,7.105427357601002e-15,7.105427357601002e-15]
 - *Main Data.Complex>
 -}
import Data.Complex

conv :: (Num a) => [a] -> [a] -> [a]
conv h x = [sum (zipWith (*) (drop (max (-i) 0) (reverse h)) (drop (max i 0) x)) | i <- [(-(length h - 1))..(length x - 1)]]

convFft :: (RealFloat a) => [a] -> [a] -> [a]
convFft h x
    | length x == length h = ifft (zipWith (*) (fft h) (fft x))
    | otherwise = error "Kernel and input data must have the same length"

fft :: (RealFloat a) => [a] -> [Complex a]
fft x
    | isPowerOfTwo n = fftRaw x n 1
    | otherwise = error "FFT works only for powers of two"
    where n = length x

fftRaw :: (RealFloat a) => [a] -> Int -> Int -> [Complex a]
fftRaw _ 0 _ = []
fftRaw [] _  _ = []
fftRaw (x0:_) 1 _ = [x0 :+ 0]
fftRaw x n s = zipWith (+) x1 x2 ++ (zipWith (-) x1 x2)
    where x1 = fftRaw x (div n 2) (2 * s)
          x2 = zipWith (*) [exp (0 :+ (-2 * pi * fromIntegral k / (fromIntegral n))) | k <- [0..((div n 2) - 1)]] (fftRaw (drop s x) (div n 2) (2 * s))

ifft :: (RealFloat a) => [Complex a] -> [a]
ifft x
    | isPowerOfTwo n = [realPart v / (fromIntegral n) | v <- ifftRaw x n 1]
    | otherwise = error "IFFT works only for powers of two"
    where n = length x

ifftRaw :: (RealFloat a) => [Complex a] -> Int -> Int -> [Complex a]
ifftRaw _ 0 _ = []
ifftRaw [] _  _ = []
ifftRaw (x0:_) 1 _ = [x0]
ifftRaw x n s = zipWith (+) x1 x2 ++ (zipWith (-) x1 x2)
    where x1 = ifftRaw x (div n 2) (2 * s)
          x2 = zipWith (*) [exp (0 :+ (2 * pi * fromIntegral k / (fromIntegral n))) | k <- [0..((div n 2) - 1)]] (ifftRaw (drop s x) (div n 2) (2 * s))

isPowerOfTwo :: Int -> Bool
isPowerOfTwo 1 = True
isPowerOfTwo x
    | mod x 2 == 0 = isPowerOfTwo (div x 2)
    | otherwise = False
