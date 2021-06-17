load('data/t2.mat')
train2.pipes = (1:height(train2))'; % append a unique ID to each row
yr_range = 2005:2017;
train2(train2.age < 0,:) = [];
obs = sum(train2.observation_year==yr_range);   % same observations each year
ob_n = obs(1);  

%% Fit to Weibull model
train2_firstFailure = train2((find((train2.Failed == 1)&(train2.n_failures == 0))),:);  % Take all samples
firstFailureAge = train2_firstFailure.observation_year - train2_firstFailure.laid_year;

[param,ci] = wblfit(firstFailureAge+1e-3);
wbl_age_pred = wblpdf(0:90,param(1),param(2));   % age 0 - 90 % 1940 - 2030

%% WBL Analysis
% table of laid and total pipes by year
r = 1;
for i = min(train2.laid_year):max(train2.observation_year)
    table_temp = train2(train2.laid_year == i,:);
    count(r) = numel(unique(table_temp.pipes));
    r = r + 1;
end
cumulative_pipes = cumsum(count);
pipe_table = table([1976:2030]',[count zeros(1,13)]',[cumulative_pipes cumulative_pipes(end)*ones(1,13)]');   % Real data + Simulated data
pipe_table.Properties.VariableNames = {'year';'laid_pipes';'total_pipes'};

% Total failure probability by year
fail_pipes_wbl = table();
for i = min(pipe_table.year):max(pipe_table.year) % loop through each year
    wbl_numFailPipes = 0;
    for j = min(train2.laid_year):i
        wbl_numFailPipes = wbl_numFailPipes + wbl_age_pred(i-j+1)*cell2mat(table2cell(pipe_table(pipe_table.year==j,2)));
    end
    fail_pipes_wbl = [fail_pipes_wbl; table(wbl_numFailPipes)];
end

pipe_table = [pipe_table fail_pipes_wbl];
wbl_FailProba = pipe_table.wbl_numFailPipes./pipe_table.total_pipes;    % 1981 - 2017
pipe_table = [pipe_table table(wbl_FailProba)];

%% Actual Failure Data
r = 1;
for i = 2000:2018
    temp_table = train2_firstFailure(train2_firstFailure.observation_year == i,:);
    [fails(r),~] = size(temp_table);
    known_FP(r) = fails(r)/cell2mat(table2cell(pipe_table(pipe_table.year == i,3)));     % failed pipes/total pipes  
    r = r+1;
end
% write to big table
known_numFailPipes = [NaN*ones(1,numel(1976:1999)) fails NaN*ones(1,numel(2019:2030))]';
known_FailProba = [NaN*ones(1,numel(1976:1999)) known_FP NaN*ones(1,numel(2019:2030))]';
pipe_table = [pipe_table table(known_numFailPipes, known_FailProba)];

%% plotting part
figure
plot(1976:2030,wbl_FailProba,'LineWidth',2)
hold on
plot(2000:2018,known_FP,'r*','MarkerSize',12)
