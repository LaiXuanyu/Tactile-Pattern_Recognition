%% Process data
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

%% Combine variables
dataMatrix = [F1pac, F1pdc, F1tac];
% Standardizing the data (z-score normalization)
standardizedData = zscore(dataMatrix);

%% B.1.a
% Calculate the covariance matrix
covMatrix = cov(standardizedData);

% Compute eigenvalues and eigenvectors
[eigenvectors, eigenvalues_matrix] = eig(covMatrix);

% Extracting the eigenvalues from the diagonal matrix
eigenvalues = diag(eigenvalues_matrix);
% Sort the eigenvalues and corresponding eigenvectors
[eigenvalues, sortIdx] = sort(eigenvalues, 'descend');
eigenvectors = eigenvectors(:, sortIdx);
% Display the sorted eigenvalues
disp('Eigenvalues:');
disp(eigenvalues);

% Display the corresponding eigenvectors
disp('Eigenvectors:');
disp(eigenvectors);

%% B.1.b

% Create labels for materials
materialTypes = {'Acrylic', 'BlackFoam', 'CarSponge', 'FlourSack', 'KitchenSponge', 'SteelVase'};
numPointsPerMaterial = 10; % Number of data points for each material
materialTypeAll = repelem(materialTypes, numPointsPerMaterial); % Repeat 10 times for each material
% Standardize Data
dataMatrix = [F1pac, F1pdc, F1tac];
standardizedData = zscore(dataMatrix);

% PCA
[coeff, score, latent] = pca(standardizedData);
% Draw 3D graph
figure;
colors = lines(numel(materialTypes)); % Generate color for each material
for i = 1:numel(materialTypes)
    idx = strcmp(materialTypeAll, materialTypes{i});
    scatter3(standardizedData(idx,1), standardizedData(idx,2), standardizedData(idx,3), 'fill', 'MarkerEdgeColor', colors(i,:));
    hold on;
end
% Set axis labels and title
xlabel('Standardized F1pac');
ylabel('Standardized F1pdc');
zlabel('Standardized F1tac');
title('3D Plot of Standardized Data with Principal Components');
legend(materialTypes, 'Location', 'bestoutside');
% Data center point
meanData = mean(standardizedData);

% Draw the three most important PC vectors using different colors
vectorLength = sqrt(latent); % Adjust vector length for visibility
quiver3(meanData(1), meanData(2), meanData(3), vectorLength(1) * coeff(1,1), vectorLength(1) * coeff(2,1), vectorLength(1) * coeff(3,1), 'r', 'LineWidth', 2);
quiver3(meanData(1), meanData(2), meanData(3), vectorLength(2) * coeff(1,2), vectorLength(2) * coeff(2,2), vectorLength(2) * coeff(3,2), 'g', 'LineWidth', 2);
quiver3(meanData(1), meanData(2), meanData(3), vectorLength(3) * coeff(1,3), vectorLength(3) * coeff(2,3), vectorLength(3) * coeff(3,3), 'b', 'LineWidth', 2);


% Calculate the scaled components of each PC vector
vector1 = vectorLength(1) * coeff(:,1);
vector2 = vectorLength(2) * coeff(:,2);
vector3 = vectorLength(3) * coeff(:,3);

% Sum the components
sumVector = vector1 + vector2 + vector3;

% Compute the average vector
averageVector = sumVector / 3;

% For visualization, you might want to normalize this average vector
averageVectorNormalized = averageVector / norm(averageVector) * mean(vectorLength); % Scale it similarly for comparison

% Now, you can visualize this average vector using quiver3, if desired
quiver3(meanData(1), meanData(2), meanData(3), averageVectorNormalized(1), averageVectorNormalized(2), averageVectorNormalized(3), 'k', 'LineWidth', 2, 'AutoScale', 'off');




% Create dummy objects for the legend's PC parts
h1 = plot3(NaN,NaN,NaN,'r', 'LineWidth', 2);
h2 = plot3(NaN,NaN,NaN,'g', 'LineWidth', 2);
h3 = plot3(NaN,NaN,NaN,'b', 'LineWidth', 2);

% Create the legend by combining scatter plot objects and dummy objects
legend( [materialTypes, {'PC1', 'PC2', 'PC3'}], 'Location', 'bestoutside');

hold off;

%% B.1.c
reducedData = score(:, 1:2);
% Visualize
uniqueMaterials = unique(materialTypeAll); % Get all unique type of materials
colors = lines(numel(uniqueMaterials)); % Generate color for each material

% Unique materials and colors
uniqueMaterials = unique(materialTypeAll); 
colors = lines(numel(uniqueMaterials));

% Create a figure
figure;
hold on;

% Scatter plot for each material type
for i = 1:numel(uniqueMaterials)
    idx = strcmp(materialTypeAll, uniqueMaterials{i});
    scatter(reducedData(idx, 1), reducedData(idx, 2),'fill', 'MarkerEdgeColor', colors(i, :));
end

% Adding principal component vectors
% Scale factor for displaying the eigenvectors
scale = max(abs(reducedData(:))) / max(abs(coeff(:)));
quiver(0, 0, scale * coeff(1, 1), scale * coeff(2, 1), 'k', 'LineWidth', 1.5);
quiver(0, 0, scale * coeff(1, 2), scale * coeff(2, 2), 'k', 'LineWidth', 1.5);

% Labels and title
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('2D PCA of Data with Principal Components');
legend(uniqueMaterials);
hold off;

%% B.1.d
% Create a new figure
figure;

% Set colors
colors = lines(numel(uniqueMaterials));

% Loop through Principal Components
for pcIndex = 1:3
    % Create subplot
    subplot(3, 1, pcIndex);
    hold on;
    
    % Get score
    pcScores = score(:, pcIndex);
    
    % Loop through all materials and plot them on the current PC
    for i = 1:numel(uniqueMaterials)
        idx = strcmp(materialTypeAll, uniqueMaterials{i});
        scatter(pcScores(idx), zeros(size(pcScores(idx))), 36, colors(i, :), 'filled');
    end
    
    % Set title and legends
    xlabel(['PC' num2str(pcIndex)]);
    if pcIndex == 1
        title('Distribution Along Principal Components');
        legend(uniqueMaterials, 'Location', 'bestoutside');
    end  
    set(gca, 'YTick', []);
    set(gca, 'YColor', 'none');
    hold off;
end


%% Process data
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


%% B.2.a

% Perform PCA on the data
[coeff, score, latent, tsquared, explained] = pca(aggregatedData');

% Scree plot of the variances
figure;
plot(explained, 'o-');
title('Scree Plot of Principal Components');
xlabel('Principal Component');
ylabel('Variance Explained (%)');
grid on;

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









