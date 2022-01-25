function features = imageDatastoreReaderCNN(datastore)
% Example of using an image datastore.

net = alexnet;
layer = 'fc7';
layerSize = 4096;
nImages = numel(datastore.Files);

features = zeros(nImages, layerSize); 
row = 1;
for i = 1:nImages
    [img, fileinfo] = readimage(datastore, i);
    % fileinfo struct with filename and another field.
    fprintf('Processing %s\n', fileinfo.Filename);

    featureVector = activations(net,img,layer,'OutputAs','rows');
    features(row,:) = featureVector;
    row = row + 1;
end
