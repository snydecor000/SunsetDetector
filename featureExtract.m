function featureVector = featureExtract(img, numBlocks)
%   This function pulls the LST means and standard deviations from an image
%   based on the number of blocks used for a grid
    img = rgb2lst(img);
    
    imgRow = size(img, 1);
    imgCol = size(img, 2);
    
    blockWidth = floor(imgCol/numBlocks);
    blockHeight = floor(imgRow/numBlocks);
    
    for r = 1:numBlocks
        for c = 1:numBlocks
        startingRow = (r-1)*blockHeight + 1;
        startingCol = (c-1)*blockWidth + 1;
        
        endRow = blockHeight + startingRow - 1;
        endCol = blockWidth + startingCol - 1;

        firstBlockPixels = img(startingRow:endRow, startingCol:endCol, :); 

        firstBlockMeanL = mean(firstBlockPixels(:,:,1), 'all');
        firstBlockMeanS = mean(firstBlockPixels(:,:,2), 'all');
        firstBlockMeanT = mean(firstBlockPixels(:,:,3), 'all');

        firstBlockStdevL = std(firstBlockPixels(:,:,1), 0 , 'all');
        firstBlockStdevS = std(firstBlockPixels(:,:,2), 0 , 'all');
        firstBlockStdevT = std(firstBlockPixels(:,:,3), 0 , 'all');
        
        if r ==1 && c == 1
            featureVector = [firstBlockMeanL, firstBlockStdevL, firstBlockMeanS, firstBlockStdevS, firstBlockMeanT, firstBlockStdevT];
        else
            featureVector = [featureVector, firstBlockMeanL, firstBlockStdevL, firstBlockMeanS, firstBlockStdevS, firstBlockMeanT, firstBlockStdevT];
        end
        end
    end
    
end
