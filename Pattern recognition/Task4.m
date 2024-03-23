%% Processing data from F1_PVT
% Load the data
load('F1_PVT.mat');

% Initialize arrays to store combined data for each feature
F1pac = [];
F1pdc = [];
F1tac = [];

% Names of the materials in newData
materials = fieldnames(newData);

% Loop through each material
for i = 1:length(materials)
    material = materials{i};

    % Loop through each trial for the material
    for j = 1:10
        trialName = sprintf('Trial%d', j); % Constructing the trial field name
        trialData = newData.(material).(trialName);

        % Combine data for each feature across all materials and trials
        F1pac = [F1pac; trialData.F1pac];
        F1pdc = [F1pdc; trialData.F1pdc];
        F1tac = [F1tac; trialData.F1tac];
    end
end

% Combine features into a single matrix for clustering
dataForClustering = [F1pac, F1pdc, F1tac];

%% D.1.a Perform K-means clustering with k = 6
k = 6;
[clusterIdx, C] = kmeans(dataForClustering, k);

% Visualise the clusters
figure;
scatter3(dataForClustering(:,1), dataForClustering(:,2), dataForClustering(:,3), 20, clusterIdx, 'filled');
hold on;
scatter3(C(:,1), C(:,2), C(:,3), 110, 'kx', 'LineWidth', 3);
title('K-means Clustering -- Euclidean Distance');
xlabel('F1pac');
ylabel('F1pdc');
zlabel('F1tac');
legend('Cluster Points', 'Centroids', 'Location', 'best');
grid on;
hold off
%% D.1.c Perform clustering using kmedoids with Manhattan distance
[clusterIdx, C] = kmedoids(dataForClustering, k, 'Distance', 'cityblock');

% Visualise the clusters
figure;
scatter3(dataForClustering(:,1), dataForClustering(:,2), dataForClustering(:,3), 20, clusterIdx, 'filled');
hold on;
scatter3(C(:,1), C(:,2), C(:,3), 110, 'kx', 'LineWidth', 3);
title('K-means Clustering -- Manhattan Distance');
xlabel('F1pac');
ylabel('F1pdc');
zlabel('F1tac');
legend('Cluster Points', 'Medoids', 'Location', 'best');
grid on;
hold off

%% Processing data (same as B.2.b)
% Load the data
data = load('F1_another.mat');

% Extract the 'extractedData' field
extractedData = data.extractedData;

% Initialize a 19x60 matrix to hold the aggregated data
aggregatedData = zeros(19, 60);

% Column index for inserting data into aggregatedData
columnIndex = 1;

% Get the field names for materials
materialNames = fieldnames(extractedData);

% Loop through each material
for i = 1:length(materialNames)
    % Get the data for this material
    materialData = extractedData.(materialNames{i});

    % Get the trial names for this material
    trialNames = fieldnames(materialData);

    % Loop through each trial
    for j = 1:length(trialNames)
        % Extract the F1Electrodes data from the trial
        trialData = materialData.(trialNames{j}).F1Electrodes;

        % Check if trialData is the expected 19x1 vector
        if isequal(size(trialData), [19, 1])
            % Insert the trial data into the aggregated matrix
            aggregatedData(:, columnIndex) = trialData;

            % Increment the column index
            columnIndex = columnIndex + 1;
        else
            error('Unexpected size of trial data in %s of %s', trialNames{j}, materialNames{i});
        end
    end
end

% Generating Labels for Each Observation
numObservations = size(aggregatedData, 2); % Initialize numObservations

labels = zeros(numObservations, 1); % Initialize labels vector
labelIndex = 1;

for i = 1:length(materialNames)
    numTrialsPerMaterial = length(fieldnames(extractedData.(materialNames{i})));
    
    labels(labelIndex:labelIndex+numTrialsPerMaterial-1) = i; % Assign label i to all trials of material i
    labelIndex = labelIndex + numTrialsPerMaterial; % Update the index for the next material
end
%% B.2.b
% Perform PCA retaining only the first three principal components
[coeff, score, ~, ~, ~] = pca(aggregatedData', 'NumComponents', 3);

materialTypes = {'Acrylic', 'BlackFoam', 'CarSponge', 'FlourSack', 'KitchenSponge', 'SteelVase'};
% Number of unique materials
uniqueMaterials = 6; % Adjust this number based on your actual data

% Generate distinct colors using 'lines'
colors = lines(uniqueMaterials);

% Creating a 3D scatter plot of the transformed data with color coding
figure;
hold on; % Ensure that the plot holds all scatter plots

for i = 1:uniqueMaterials
    start_index = (i-1)*10 + 1; % Adjust if different number of trials per material
    end_index = i*10;           % Adjust if different number of trials per material
    scatter3(score(start_index:end_index,1), score(start_index:end_index,2), score(start_index:end_index,3), 36, colors(i,:), 'filled'); % Adjust size if needed
end

hold off;

xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
title('3D Visualization using the First Three Principal Components');
legend(materialTypes);
grid on;

% Set the view for a 3D plot
view(3);

%% Splitting the PCA-processed data into Training and Test sets
rng(6); % ensure the experiment reesult are same
numObservations = size(score, 1);
numTrain = floor(0.6 * numObservations); % 60% for training

% Generate random indices for training data
randIndices = randperm(numObservations);
trainIndices = randIndices(1:numTrain);
testIndices = randIndices(numTrain+1:end);

% Split the data and labels into training and testing sets
trainData = score(trainIndices, :);
trainLabels = labels(trainIndices);
testData = score(testIndices, :);
testLabels = labels(testIndices);
%% D.2.a find the best number of tree
% identify number of trees we want to test
numTrees_values = 1:50;
accuracy_values = zeros(size(numTrees_values)); % initilization

% train model and test accuracy
for i = 1:length(numTrees_values)
    numTrees = numTrees_values(i);
    baggedModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification');
    
    % predicy label
    [predictedLabel, score] = predict(baggedModel, testData);
    predictedLabel = str2double(predictedLabel); % transfer
    
    % calculate and save accuracy
    accuracy = sum(predictedLabel == testLabels) / length(testLabels);
    accuracy_values(i) = accuracy;
end

% draw scree graph
figure;
plot(numTrees_values, accuracy_values * 100, '-o'); 
xlabel('Number of Trees in the Ensemble');
ylabel('Accuracy (%)');
title('Accuracy vs. Number of Trees in the Ensemble');
grid on;

% Applying Bagging (Bootstrap Aggregation) with Decision Trees
% Using MATLAB's TreeBagger function to create an ensemble of decision trees
numTrees = 26; % Number of trees in the ensemble
baggedModel = TreeBagger(numTrees, trainData, trainLabels, 'Method', 'classification');

%% D.2.b Visualsing data
% Visualize the first decision tree
figure;
tree1 = baggedModel.Trees{1}; % Extract the first tree from the ensemble
view(tree1, 'Mode', 'graph');

% Visualize the second decision tree
figure;
tree2 = baggedModel.Trees{2}; % Extract the second tree from the ensemble
view(tree2, 'Mode', 'graph');
%% D.2.c Evaluating the Model on the Test Set
% Predict the responses for the test set
[predictedLabels, scores] = predict(baggedModel, testData);

% Convert predictedLabels to the appropriate type if necessary (depending on how 'labels' is formatted)
predictedLabels = str2double(predictedLabels);

% Calculate the confusion matrix
[C, order] = confusionmat(testLabels, predictedLabels);

% Visualize the confusion matrix
figure;
confusionchart(C, order);

% Calculate the accuracy or other performance metrics
accuracy = sum(predictedLabels == testLabels) / length(testLabels);
fprintf('Accuracy of the bagged ensemble on the test set: %.2f%%\n', accuracy * 100);
