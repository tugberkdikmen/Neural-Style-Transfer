I = imread('1.tif');
grayImage = rgb2gray(I);

gaborArray = gabor([4 8],[0 90]);

Apply filters to input image.
gaborMag = imgaborfilt(grayImage,gaborArray);

figure
subplot(2,2,1);
for p = 1:4
    subplot(2,2,p)
    imshow(gaborMag(:,:,p),[]);
    theta = gaborArray(p).Orientation;
    lambda = gaborArray(p).Wavelength;
    title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
end