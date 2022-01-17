function lstImg = rgb2lst(rgbImg)
%RGB2LST converts an RGB image into an LST image and returns it.
%   Given an RGB image (uint8 3 channel image), the L (Intensity),
%   S (Red vs Blue), and T (Green vs Magenta) are calculated and returned
%   in a uint8 3 channel image
rgbImg = double(rgbImg);
R = rgbImg(:,:,1);
G = rgbImg(:,:,2);
B = rgbImg(:,:,3);

L = (R+G+B)/3;
S = R-B;
T = R-2*G+B;

lstImg = zeros(size(rgbImg));
lstImg(:,:,1) = L;
lstImg(:,:,2) = S;
lstImg(:,:,3) = T;
lstImg = uint8(lstImg);
