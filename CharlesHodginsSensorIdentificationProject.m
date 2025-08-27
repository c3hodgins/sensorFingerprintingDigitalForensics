%% Charles Hodgins - Sensor Fingerprint Analysis Project
function camera_identification()
    try
        load('K.mat', 'K');
        K_processed = preprocess_prnu(K);
        figure(1);
        K_gray = 0.3*K_processed(:,:,1) + 0.6*K_processed(:,:,2) + 0.1*K_processed(:,:,3);
        imshow(mat2gray(K(1:256,1:256,:)-K_processed(1:256,1:256,:)))
    catch ME
        fprintf('Error loading PRNU: %s\n', ME.message);
        return;
    end

    image_dir = 'All_34_images';
    image_files = dir(fullfile(image_dir, '*.jpg'));
    pce_results = cell(length(image_files), 2);
    
    for i = 1:length(image_files)
        try
            img_path = fullfile(image_dir, image_files(i).name);
            fprintf('Processing %s...\n', image_files(i).name);
            img = double(imread(img_path)) / 255.0; 
            
            if ~validate_conversion(img)
                continue;
            end
            
            W = get_noise_residual(img);
            
            [pce, ncc] = compute_pce(W, K_gray, 5);
            pce_results{i,1} = image_files(i).name;
            pce_results{i,2} = pce;
            
            if pce > 60
                figure(3);
                mesh(ncc);
                title(sprintf('NCC for %s (PCE=%.1f)', image_files(i).name, pce));
            end
            
        catch ME
            fprintf('Error processing %s: %s\n', image_files(i).name, ME.message);
            continue;
        end
    end
    
    fprintf('\nCamera Identification Results:\n');
    [~, idx] = sort(cell2mat(pce_results(:,2)), 'descend');
    for i = 1:length(idx)
        fprintf('%s: %.1f\n', pce_results{idx(i),1}, pce_results{idx(i),2});
    end
    
    figure(2);
    semilogy(cell2mat(pce_results(:,2)), 'o-');
    hold on;
    yline(60, 'r--', 'Threshold (PCE=60)');
    xlabel('Image Index');
    ylabel('PCE (log scale)');
    title('Canon 6D Camera Identification');
    grid on;
end
%%
function K_processed = preprocess_prnu(K)
    K_processed = K;
    for c = 1:3
        col_means = mean(K_processed(:,:,c), 1);
        K_processed(:,:,c) = K_processed(:,:,c) - col_means;
        
        row_means = mean(K_processed(:,:,c), 2);
        K_processed(:,:,c) = K_processed(:,:,c) - row_means;
    end
end

function valid = validate_conversion(img)
    valid = true;
    if ~isa(img, 'double')
        fprintf('CRITICAL: Image is %s, not double!\n', class(img));
        valid = false;
    end
    if max(img(:)) > 1 || min(img(:)) < 0
        fprintf('WARNING: Image range [%.2f, %.2f]\n', min(img(:)), max(img(:)));
    end
end

function W = get_noise_residual(img)
    W = double(zeros(size(img)));
    for c = 1:3
        channel = img(:,:,c);
        W(:,:,c) = channel - wiener2(channel);
    end
    W = 0.3*W(:,:,1) + 0.6*W(:,:,2) + 0.1*W(:,:,3);
end

function [pce, ncc] = compute_pce(W, K, exclude_region)
   
    pad_rows = size(K,1) - size(W,1);
    pad_cols = size(K,2) - size(W,2);
    
    if pad_rows > 0 || pad_cols > 0
        W = padarray(W, [max(0,pad_rows), max(0,pad_cols)], 0, 'post');
    else
        W = W(1:size(K,1), 1:size(K,2));
    end
    ncc = crosscorr2(W, K);  
    
    [peak_vals, peak_rows] = max(ncc);
    [peak_val, peak_col] = max(peak_vals);
    peak_row = peak_rows(peak_col);
    
    [m, n] = size(ncc);
    [X, Y] = meshgrid(1:n, 1:m);
    exclusion_mask = (abs(X-peak_col) <= exclude_region/2) & ...
                    (abs(Y-peak_row) <= exclude_region/2);
    
    ncc_squared = ncc.^2;
    ncc_squared(exclusion_mask) = 0;  
    energy = sum(ncc_squared(:)) / (m*n - sum(exclusion_mask(:)));
    
    pce = peak_val^2 / (energy + eps);  
end

function ret = crosscorr2(array1, array2)
% function ret = crosscor2(array1, array2)
% Computes 2D crosscorrelation of 2D arrays
% Function returns DOUBLE type 2D array ret!
array1 = double(array1);
array2 = double(array2);

if ~(size(array1,1)==size(array2,1) && size(array1,2)==size(array2,2))
    fprintf('  The array dimensions do not match.\n')
    ret = 0;
else
    array1 = array1 - mean(array1(:));
    array2 = array2 - mean(array2(:));

    %%%%%%%%%%%%%%% End of filtering
    tilted_array2 = fliplr(array2);
    tilted_array2 = flipud(tilted_array2);
    TA = fft2(tilted_array2);
    FA = fft2(array1);
    AC = FA .* TA;
    normalizer = sqrt(sum(array1(:).^2)*sum(array2(:).^2));

    if normalizer==0,
        ret = 0;
    else
        ret = real(ifft2(AC))/normalizer;
    end
end
end