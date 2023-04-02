function [resulttab, indresults, solresults, sizeresults, hyperparam] = RunFS(x,y, hyperparam)
% Implementation for Feature Selection and Ensemble Feature Selection

%% Hyperparameters & Selections

% Detailed information (yes/no)
hyperparam.info = 0; % 1: additional information, 0 no additional information

% Method used
hyperparam.method = 'KNN'; % classifier used

% Hyperparameters for internal and external CV
hyperparam.runs = 5; % number of runs (of the entire process with external and internal CV)
hyperparam.foldsexternalCV = 5; % number of folds (first CV split)
hyperparam.foldsinternalCV = 5; % number of folds (second CV split)

% Evaluation Criterion
hyperparam.fobj = @ObjectiveFunction; % objective function
hyperparam.objtype = 2; % 1 for error, 2 for 0.99 * Error + 0.01 * No retained features


% Hyperparmaeters for Ensembles
hyperparam.ensemblesize = 3;  % number of base selectors
hyperparam.ensemblebootstrapshare = 0.9; % share of data for bootstrap
hyperparam.ensemblebootstrapreplace = false; % share of data for bootstrap
hyperparam.ensembleaggDataPerturb = {'intersection','share/number','share/number','share/number','share/number','share/number','share/number','union'}; % 'intersection', 'union', 'maxfrequent', 'share/number' [if 1 or larger --> number]
hyperparam.ensembleaggDataPerturbNumber = [NaN, 1, 2, 4, 8, 10, 20, NaN]; % NaN for methods  that do not require a specification; 0.1 --> 10%, 10 --> 10 features

% Hyperparameters for classification methods
hyperparam.methodKNNkval = 5; % kvalue

% (Binary) Standard Genetic Algorithm (SGA)
hyperparam.methodSGAPopsize = 30; % Population Size
hyperparam.methodSGAMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)
hyperparam.methodSGACrossover = 'single'; % 'single' for one-point cross-over, 'double' for two-point crossover, 'uniform' for uniform crossover
hyperparam.methodSGACrossoverProp = 0.85; % Crossover Probability
hyperparam.methodSGAMutation = 'bitflip'; % 'bitflip' for uniform crossover, 'real01' for real-valued number in [0,1]
hyperparam.methodSGAMutationProp = 0.10; % Mutation Probability

% Binary Genetic Algorithm (BGA)
hyperparam.methodBGAPopsize = 30; % Population Size
hyperparam.methodBGAMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)
hyperparam.methodBGACrossover = 'single'; % 'single' for one-point cross-over, 'double' for two-point crossover, 'uniform' for uniform crossover
hyperparam.methodBGACrossoverProp = 0.85; % Crossover Probability
hyperparam.methodBGAMutation = 'bitflip'; % 'bitflip' for uniform crossover, 'real01' for real-valued number in [0,1]
hyperparam.methodBGAMutationProp = 0.10; % Mutation Probability
hyperparam.methodBGAElitismShare = 0.10; % Share of Elitism

% (real-valued) Genetic Algorithm (GA)
hyperparam.methodGAPopsize = 30; % Population Size
hyperparam.methodGAMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)
hyperparam.methodGACrossover = 'single'; % 'single' for one-point cross-over, 'double' for two-point crossover, 'uniform' for uniform crossover
hyperparam.methodGACrossoverProp = 0.85; % Crossover Probability
hyperparam.methodGAMutation = 'real01'; % 'bitflip' for uniform crossover, 'real01' for real-valued number in [0,1]
hyperparam.methodGAMutationProp = 0.10; % Mutation Probability
hyperparam.methodGAElitismShare = 0.10; % Share of Elitism

% (real-valued) Grey Wolf Optimization (GWO)
hyperparam.methodGWOPopsize = 30; % Population Size
hyperparam.methodGWOMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)

% (real-valued) Differential Evolution (DE)
hyperparam.methodDEPopsize = 30; % Population Size
hyperparam.methodDEMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)
hyperparam.methodDECrossover = 'uniform'; % 'single' for one-point cross-over, 'double' for two-point crossover, 'uniform' for uniform crossover
hyperparam.methodDECrossoverProp = 0.8; % Crossover Probability
hyperparam.methodDEBetaMin = 0.5; % Minimum for Scaling Factor Beta
hyperparam.methodDEBetaMax = 0.5; % Maximum for Scaling Factor Beta

% (real-valued) Particle Swarm Optimization (PSO)
hyperparam.methodPSOPopsize = 30; % Population Size
hyperparam.methodPSOMax_iteration = 100; % Maximum Number of Iterations (Stopping Criterion)
hyperparam.methodPSOVelocityMax = 1; % maximum Velocity [for FS high enough to slip membership]
hyperparam.methodPSOInertiaWeightMax = 0.9; % maximum inertia weight
hyperparam.methodPSOInertiaWeightMin = 0.2; % minimum inertia weight
hyperparam.methodPSOc1 = 2; % coefficient for personal / cognitive component
hyperparam.methodPSOc2 = 2; % coefficient for global / social component

%% User Choices
% FS Strategies
Strategylist = {'Individual Feature Selection',
    'Data Perturbation [Ensemble]'};

[indx,tf] = listdlg('PromptString',{'Select one strategy for feature selection.',...
    'Only one option can be selected at a time.',''},...
    'SelectionMode','single','ListString',Strategylist,...
    'Name','Feature Selection (Step 1)','ListSize',[300,150]);

Strategyselected = Strategylist(indx);

% FS Methods
FSlistshort = {'NoFS','SGA','GA','BGA','DE','GWO','PSO'};
FSlist = {'No Feature Selection (Benchmark)',
    'Simple Genetic Algorithm (SGA)',
    'Genetic Algorithm (GA)',
    'Binary Genetic Algorithm (BGA)',
    'Differential Evolution (DE)',
    'Grey Wolf Optimization (GWO)',
    'Particle Swarm Optimization (PSO)'};

% FS Methods Selection (depends on Strategy Selection)
switch Strategyselected{1}
    case 'Data Perturbation [Ensemble]'
        [indx,tf] = listdlg('PromptString',{'Select one feature selection algorithms.',...
    'Only one can be selected at a time.',''},...
    'SelectionMode','single','ListString',FSlist,...
    'Name','Feature Selection (Step 2)','ListSize',[300,250]);
        
    otherwise
        [indx,tf] = listdlg('PromptString',{'Select one or multiple feature selection algorithms.',...
    'One or multiple can be selected at a time.',''},...
    'SelectionMode','multiple','ListString',FSlist,...
    'Name','Feature Selection (Step 2)','ListSize',[300,250]);
end


%% Prepare list of algorithms to run
switch Strategyselected{1}
    case 'Individual Feature Selection'
        FSselected = FSlistshort(indx);
        nomethods = size(indx,2); % Number of methods run
    case 'Data Perturbation [Ensemble]'
        FSselected = FSlistshort(repmat(indx,1,hyperparam.ensemblesize));  
        nomethods = hyperparam.ensemblesize; % Number of methods run (in ensemble) 
        
        % Pre-process results vector for the ensemble
        ensolresults = cell(hyperparam.foldsexternalCV * hyperparam.runs, size(hyperparam.ensembleaggDataPerturb,2)); % ensemble subsets: all folds (for all runs) x aggregation methods used
        enindresults = zeros(hyperparam.foldsexternalCV * hyperparam.runs, size(hyperparam.ensembleaggDataPerturb,2)); % ensemble performance: all folds (for all runs) x aggregation methods used
        ensizeresults = zeros(hyperparam.foldsexternalCV * hyperparam.runs, size(hyperparam.ensembleaggDataPerturb,2)); % ensemble size: all folds (for all runs) x aggregation methods used
        enfitnessresults = zeros(hyperparam.foldsexternalCV * hyperparam.runs, size(hyperparam.ensembleaggDataPerturb,2)); % ensemble fitness: all folds (for all runs) x aggregation methods used
end

%% Information to the user
clc
disp(horzcat('Strategy: Selected Strategy: ',Strategyselected{:}));
disp(horzcat('Methods: Selected Methods: ',strjoin(FSselected(:),', ')));

%% Pre-processing of results
solresults = cell(hyperparam.foldsexternalCV * hyperparam.runs, nomethods); % best solution in each fold
indresults = zeros(hyperparam.foldsexternalCV * hyperparam.runs, nomethods); % pre-process results of methods
sizeresults = zeros(hyperparam.foldsexternalCV * hyperparam.runs, nomethods); % pre-process results of methods
fitnessresults = cell(hyperparam.foldsexternalCV * hyperparam.runs, nomethods); % fitness development in each fold

%% Main loop over all methods
tic
for i = 1 : hyperparam.runs

    % Data Division (in each run)
    excrossval = cvpartition(y,'KFold',hyperparam.foldsexternalCV,'Stratify',true);
    % excrossval = cvpartition(n,'Leaveout')

    for ex = 1 : hyperparam.foldsexternalCV
        switch Strategyselected{1}
            case 'Data Perturbation [Ensemble]'

                % Training Data (External CV)
                dataTrainEx = x(excrossval.training(ex),:);
                classTrainEx = y(excrossval.training(ex),:);

                % Boostrap of the external CV Training Data
                [dataTrain,idxBootStrap] = datasample(dataTrainEx,round(hyperparam.ensemblebootstrapshare * size(dataTrainEx,1)),'Replace',hyperparam.ensemblebootstrapreplace);
                classTrain = classTrainEx(idxBootStrap);

                % Test Data (External CV)
                dataTest = x(excrossval.test(ex),:);
                classTest = y(excrossval.test(ex),:);

            otherwise
                % Training Data (External CV)
                dataTrain = x(excrossval.training(ex),:);
                classTrain = y(excrossval.training(ex),:);

                % Test Data (External CV)
                dataTest = x(excrossval.test(ex),:);
                classTest = y(excrossval.test(ex),:);                   
        end
        
        % Run selected feature selection methods (on the same training data)
        for m = 1 : nomethods
            switch FSselected{m}
                case 'NoFS'
                    disp(horzcat('###### Method ',num2str(m),': No Feature Selection [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % No Feature Selection - Benchmark model
                    solution = 1:size(x,2); % All features
                    fitnessDev = zeros(1,2); % no fitness development for No feature selection recorded

                case 'SGA'
                    disp(horzcat('###### Method ',num2str(m),': Simple Genetic Algorithm [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % (real-valued) Simple Genetic Algorithm (SGA)
                    [solution, fitness, fitnessDev] = my_SGA(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]

                case 'GA'
                    disp(horzcat('###### Method ',num2str(m),': Genetic Algorithm [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % (real-valued) Genetic Algorithm (GA)
                    [solution, fitness, fitnessDev] = my_GA(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]
                
                case 'BGA'
                    disp(horzcat('###### Method ',num2str(m),': Binary Genetic Algorithm [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % Binary Genetic Algorithm (BGA)
                    [solution, fitness, fitnessDev] = my_BGA(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]

                case 'DE'
                    disp(horzcat('###### Method ',num2str(m),': Differential Evolution [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % (real-valued) Differential Evolution (DE)
                    [solution, fitness, fitnessDev] = my_DE(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]

                case 'GWO'
                    disp(horzcat('###### Method ',num2str(m),': Grey Wolf Optimization [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % (real-valued) Grey Wolf Optimization (GWO)
                    [solution, fitness, fitnessDev] = my_GWO(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]

                case 'PSO'
                    disp(horzcat('###### Method ',num2str(m),': Particle Swarm Optimization [Run ', num2str(i), ' out of ', num2str(hyperparam.runs),' - Fold ', num2str(ex),' out of ',num2str(hyperparam.foldsexternalCV),' [Duration: ', num2str(round(toc/60,1)),' Min] ######'))
                    % Particle Swarm Optimization (PSO)
                    [solution, fitness, fitnessDev] = my_PSO(dataTrain, classTrain, hyperparam); % (1) x, (2) y, (3) method, (4) info, (5) objtype [fitness function]

                otherwise
                    disp('other value')
            end


            % Determine Performance on the test set 
            [testerror, nofeatures] = EvalFunction(dataTrain(:,solution), classTrain, dataTest(:,solution), classTest, hyperparam);

            % Record test error, number of features retained, and solutions
            indresults((ex +(i-1) * hyperparam.foldsexternalCV),m) = testerror;
            solresults{(ex +(i-1) * hyperparam.foldsexternalCV),m} = solution;
            sizeresults((ex +(i-1) * hyperparam.foldsexternalCV),m) = nofeatures;
            fitnessresults{(ex +(i-1) * hyperparam.foldsexternalCV),m} = fitnessDev(:,2);

            %% Aggregation [ONLY for Ensembles]

            % After all methods in ensemble have been run and a subset was obtained for each
            if m == nomethods
                switch Strategyselected{1}
                    case 'Data Perturbation [Ensemble]'
                        for a = 1 : size(hyperparam.ensembleaggDataPerturb,2) % for each aggregation method
                            % Determine Aggregated Subset
                            [aggsolution, ~] = AggFunction(solresults((ex +(i-1) * hyperparam.foldsexternalCV),:), hyperparam.ensembleaggDataPerturb{a}, hyperparam.ensembleaggDataPerturbNumber(a), size(dataTrain,2)); % (1) Subsets (cell), (2) runs/folds, (3) aggregation method e.g. "union", (4) number/share of most frequent subsets (5) number of features in the entire data set

                            % Determine Fitness on the validation data set(s) for comparison to individual methods
                            tempaggsolution = zeros(1,size(x,2));
                            tempaggsolution(aggsolution) = 1;
                            enfitnessresults((ex +(i-1) * hyperparam.foldsexternalCV),a) = hyperparam.fobj(tempaggsolution, dataTrain, classTrain, hyperparam); % Individual, data (no class), class labels, method (e.g. 'KNN'), objtype (e.g. 1)                           
                            
                            % Determine Performance on the test set
                            [testerror,nofeatures] = EvalFunction(dataTrain(:,aggsolution), classTrain, dataTest(:,aggsolution), classTest, hyperparam);
                            enindresults((ex +(i-1) * hyperparam.foldsexternalCV),a) = testerror;
                            ensolresults{(ex +(i-1) * hyperparam.foldsexternalCV),a} = aggsolution;
                            ensizeresults((ex +(i-1) * hyperparam.foldsexternalCV),a) = nofeatures;
                        end
                end
            end

            % ALL of the same get aggregated (potentially second aggregation)
            %%%%%%%%% TO BE DONE %%%%%%%%%

        end % methods (m)
        

        %% OPTIONAL: information on the completion
        if hyperparam.info == 1
            disp(horzcat(num2str(FSselected{m}),'[Run ',num2str(i), ' out of ', num2str(hyperparam.runs),']',' completed [Duration: ', num2str(round(toc/60,1)),' Min]'));
        end
       
    end % folds (ex)
end % runs (i)
toc

%% Average Fitness Values for each method
% pre-processing the results
meanfitnessresults = zeros(size(fitnessresults{1,1},1),size(fitnessresults,2));
for f = 1 : size(fitnessresults,2)
    meanfitnessresults(:,f) = mean(([fitnessresults{:,f}]),2);
end

% plot(1:size(fitnessresults{1,1},1),meanfitnessresults(:,1),'r-')

%% Determine Stability of the Results (for each run)
switch Strategyselected{1}
    case 'Individual Feature Selection'
        [ASMmean, ASMstd] = StabilityASM(solresults, size(x,2), hyperparam.runs, hyperparam.foldsexternalCV); % subsets, NumFeatures, Number of Runs, Number of folds
    
    case 'Data Perturbation [Ensemble]'
        [ASMmean, ASMstd] = StabilityASM(ensolresults, size(x,2), hyperparam.runs, hyperparam.foldsexternalCV); % subsets, NumFeatures, Number of Runs, Number of folds
        
end

%% Save results in a table

switch Strategyselected{1}
    case 'Individual Feature Selection'
        for m = 1 : nomethods
            if m == 1
                % Recording results statistics
                resulttab = table(convertCharsToStrings(FSselected(m)), mean(indresults(:,m)), std(indresults(:,m)),mean(sizeresults(:,m)), std(sizeresults(:,m)), ...
                        ASMmean(m),ASMstd(m),'VariableNames',{'Name', 'AvgTestErr', 'StdTestErr', 'AvgNoFeat', 'StdNoFeat','ASMmean','ASMstd'});
            else
                % Recording result statistics
                resulttab = [resulttab; table(convertCharsToStrings(FSselected(m)), mean(indresults(:,m)), std(indresults(:,m)),mean(sizeresults(:,m)), std(sizeresults(:,m)), ...
                    ASMmean(m),ASMstd(m),'VariableNames',{'Name', 'AvgTestErr', 'StdTestErr', 'AvgNoFeat', 'StdNoFeat','ASMmean','ASMstd'})];
            end
        end       
        
    case 'Data Perturbation [Ensemble]'
        % Recording results statistics (for each aggregation method)
        for m = 1 : size(hyperparam.ensembleaggDataPerturb,2)
            if m == 1
                % For first aggregation method
                resulttab = table(convertCharsToStrings(horzcat('Ensemble ',FSselected{1},' (',hyperparam.ensembleaggDataPerturb{m},')')), mean(enindresults(:,m)), std(enindresults(:,m)),...
                    mean(ensizeresults(:,m)), std(ensizeresults(:,m)), ASMmean(m), ASMstd(m),'VariableNames',{'Name', 'AvgTestErr', 'StdTestErr', 'AvgNoFeat', 'StdNoFeat','ASMmean','ASMstd'});
            else
                % For all other aggregation methods
                resulttab = [resulttab; table(convertCharsToStrings(horzcat('Ensemble ',FSselected{1},' (',hyperparam.ensembleaggDataPerturb{m},')')), mean(enindresults(:,m)), std(enindresults(:,m)),...
                    mean(ensizeresults(:,m)), std(ensizeresults(:,m)), ASMmean(m), ASMstd(m),'VariableNames',{'Name', 'AvgTestErr', 'StdTestErr', 'AvgNoFeat', 'StdNoFeat','ASMmean','ASMstd'})];
            end
        end                  
        
end


end % FUNCTION end