function [ASMmean, ASMstd, ASMall] = StabilityASM(solresults, NumFeatures, runs, CVfolds)
% Determines the "Adjusted Stability Measure" (ASM) for feature subsets
% (over multiple folds) for each runs

% if no number of separate runs specified, one run is assumed
if nargin < 3
    runs = 1;
    CVfolds = size(solresults,1);
end

% Preprocessing of the similarity of two feature sets
SA = zeros((CVfolds*(CVfolds-1))/2, size(solresults,2), runs); % Pairwise comparisons [C * (C-1))/2 ] x Methods x Runs

% for each run and method determine the pairwise similarities of subsets for different folds (for the same method)
for i = 1 : runs
    subsets = solresults((CVfolds * (i-1) + 1):(CVfolds *i),:);
    for j = 1 : size(solresults,2) % for each method
        counter = 1;
        for ex = 1 : CVfolds-1 % for each fold (up to the one before the last)
            for ex2 = (ex+1) : CVfolds % for each other fold   
                %% ASM calculation for a specific classifier and feature selection method
                % If function to avoid NA (since if numerator is zero, the denominator is as well - which results in an NA due to division by zero
                if ( length(intersect(cell2mat(subsets(ex,j)),cell2mat(subsets(ex2,j)))) - length(cell2mat(subsets(ex,j))) * length(cell2mat(subsets(ex2,j))) / NumFeatures ) == 0
                    SA(counter, j, i) = 0;
                else
                SA(counter, j, i) = ( length(intersect(cell2mat(subsets(ex,j)),cell2mat(subsets(ex2,j)))) - ( length(cell2mat(subsets(ex,j))) * length(cell2mat(subsets(ex2,j))) ) / NumFeatures ) / ...
                    ( min(length(cell2mat(subsets(ex,j))), length(cell2mat(subsets(ex2,j)))) - max(0, length(cell2mat(subsets(ex,j))) + length(cell2mat(subsets(ex2,j))) - NumFeatures) );  
                end
                counter = counter + 1;
            end
        end  
    end
end

% Determine average similarity over pairs of subsets (for each run and method)
if runs > 1 & size(solresults,2) > 1 % if more than 1 run and more than 1 method
    ASMmean = mean(squeeze(mean(SA,1))',1); % 1 x methods (mean over runs)
    ASMstd = std(squeeze(mean(SA,1))',1); % 1 x methods (mean over runs)
    ASMall = squeeze(mean(SA,1))'; 
    
elseif runs > 1 & size(solresults,2) == 1 % if more than 1 run and more than 1 method [special case for squeeze and transpose]
    ASMmean = mean(squeeze(mean(SA,1)),1); % 1 x methods (mean over runs)
    ASMstd = std(squeeze(mean(SA,1)),1); % 1 x methods (mean over runs)
    ASMall = squeeze(mean(SA,1))';
    
else % if only 1 run
    ASMmean = mean(SA,1); % 1 x methods [only for single run, just ASM value not mean of ASVM values]
    ASMstd = zeros(1,size(ASMmean,2)); % no standard deviation
    ASMall = mean(SA,1);
end

end

