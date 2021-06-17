function [auc_pipe,auc_len, per_pipe,per_len, ROC]=getAUCROC(filename)   
    %%Given a data file with the results of the either a survival analysis or a Random Forest, Gradient boost regression or baseline analysis, This function calculates the per pipe number ROC as well as the per pipe length ROC. 
    %It also calculates the area under the curve(AUC) values for these
    %ROCs. Final ROCs are also plotted.
    data = readtable(filename);
    [ROC,per_pipe,per_len] = calculateROC(data,filename);
    
    figure;
    plot(per_pipe,ROC,'linewidth',2);
    title('Per pipe number ROC')
    xlabel('Pipe number percentage') 
    ylabel('ROC') 
    
    figure    
    plot(per_len,ROC,'linewidth',2)
    title('Per pipe length ROC')
    xlabel('Pipe length percentage') 
    ylabel('ROC') 
    
    auc_pipe = trapz(per_pipe,ROC)*1e-4;
    auc_len = trapz(per_len,ROC)*1e-4;    
    
end

