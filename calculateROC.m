function [ROC,pipe_per,pipe_len] = calculateROC(data,fname)
    %Calculate the per pipe number(pipe_per) and per length(pipe_len) ROC values for the given dataset
    %Achieved by sorting the pipes based on failure probability and then taking the cumulative sum of the total failures, the number of pipes and the
    %length of pipes in the sorted order of the pipes.
    %Returns the cumulative sums along with the calculated ROC. All quantities presented as percentages.
   
    data = sortrows(data,find(strcmpi(data.Properties.VariableNames,'y')),'descend');
    
    failSum = cumsum(data.Failed);
    ROC = failSum./max(failSum)*100;
    pipe_per = (1:length(failSum))/length(failSum)*100; % Each pipe as a percentage
    lenSum = cumsum(data.length);
    pipe_len = lenSum./max(lenSum)*100; 

end