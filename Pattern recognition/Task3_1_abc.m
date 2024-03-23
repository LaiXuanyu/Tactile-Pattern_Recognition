% Load the data
load('F1_PVT.mat');

% Initialize arrays to store combined data for each feature and labels
F1pac = [];
F1pdc = [];
F1tac = [];
labels = [];  % Array to store labels

% Names of the materials in newData
materials = {'BlackFoam', 'CarSponge'};  % Use cell array for material names

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

        % Add labels
        labels = [labels; repmat({material}, length(trialData.F1pac), 1)];
    end
end

% Combine all feature data into a single matrix for standardization
all_features = [F1pac F1pdc F1tac];

% Calculate the mean and standard deviation for each feature
means = mean(all_features, 1);
stds = std(all_features, 0, 1);

% Apply standardization to each feature
F1pac_standardized = (F1pac - means(1)) / stds(1);
F1pdc_standardized = (F1pdc - means(2)) / stds(2);
F1tac_standardized = (F1tac - means(3)) / stds(3);
%% 

% C.1.a -----------------------------------------------------------------
% combine the feature 
data_PV = [F1pdc_standardized, F1pac_standardized];    % Pressure vs Vibration
data_PT = [F1pdc_standardized, F1tac_standardized];    % Pressure vs Temperature
data_TV = [F1tac_standardized, F1pac_standardized];    % Temperature vs Vibration


% Pressure vs Vibration
axisname_PV = {'Pressure' , 'Vibration'};
figure;
plotLDA_2feature(data_PV, labels, axisname_PV, 'Pressure vs Vibration');

% Pressure vs Temperature
axisname_PT = {'Pressure' , 'Temperature'};
figure;
plotLDA_2feature(data_PT, labels, axisname_PT,'Pressure vs Temperature');

% Temperature vs Vibration
axisname_TV = {'Temperature' , 'Vibration'};
figure;
plotLDA_2feature(data_TV, labels, axisname_TV,'Temperature vs Vibration');
%% 

% C.1.b -----------------------------------------------------------------
% combine the feature 
data_PVT = [F1pdc_standardized, F1pac_standardized, F1tac_standardized];


% use LDA function
lda = fitcdiscr(data_PVT, labels);

% acquire the parameter of the
W = lda.Coeffs(1,2).Linear; 
b = lda.Coeffs(1,2).Const;  

% labels are str, can't be used in scatter3 function
% change it into value
uniqueLabels = unique(labels); % find unique labels
numClasses = numel(uniqueLabels); 
colors = lines(numClasses); % create color for each class

% draw plot
figure;

hold on;
for i = 1:numClasses
    % return labels row
    idx = strcmp(labels, uniqueLabels{i});
    scatter3(data_PVT(idx,1), data_PVT(idx,2), data_PVT(idx,3), 10, colors(i,:), 'filled');
end
% Create the grid plane
[xGrid, yGrid] = meshgrid(linspace(min(data_PVT(:,1)), max(data_PVT(:,1)), 50), ...
                          linspace(min(data_PVT(:,2)), max(data_PVT(:,2)), 50));
zGrid = (-W(1)*xGrid - W(2)*yGrid - b) / W(3);

% draw the plane, to make the points easy to classify, use light yellow
mesh(xGrid, yGrid, zGrid, 'EdgeColor', [250/255, 250/255, 170/255],'Linestyle','-','FaceColor', 'none');

% Set up the figure
xlabel('Pressure');
ylabel('Vibration');
zlabel('Temperature');
title('LDA on PVT Data');
legend('BlackFoam', 'CarSponge');
grid on
view(3); 
hold off;

% C.1.d -----------------------------------------------------------------



function plotLDA_2feature(data, labels, axisname, titleText)
    % Set the figure position and size
    fig = gcf;
    fig.Position = [100, 100, 500, 500];

    % Apply LDA
    lda = fitcdiscr(data, labels);

    % Plot data points
    gscatter(data(:, 1), data(:, 2), labels);
    hold on;

    % Calculate the decision boundary
    W = lda.Coeffs(1,2).Linear;
    b = lda.Coeffs(1,2).Const;

    % Set the y-axis range to [-2, 2]
    yline = linspace(-2, 2, 100);

    % Calculate x values for the decision boundary
    if abs(W(1)) > 1e-6
        xline = (-W(2) * yline - b) / W(1);
    else
        xline = mean(data(:,1)) * ones(size(yline));
    end

    % Plot the decision boundary
    plot(xline, yline, 'k--', 'LineWidth', 2);

    % Calculate the slope of the line perpendicular to the decision boundary
    if abs(W(2)) > 1e-6  % ensure w(2) not equal to zero
        slope = W(2) / W(1);
    else
        slope = 1e6; % If W(2) is close to zero, set a large slope value
    end

    % Calculate x and y values for the perpendicular line
    x_perpendicular = linspace(-2, 2, 100);
    y_perpendicular = slope * x_perpendicular;

    % Plot the line perpendicular to the decision boundary
    plot(x_perpendicular, y_perpendicular, 'g-', 'LineWidth', 2);

    % Set the figure settings
    title(titleText);
    xlabel(axisname{1});
    ylabel(axisname{2});
    legend(unique(labels), 'Location', 'best');

    % Ensure equal axis scaling
    axis equal;
    axis([-2 2 -2 2]);
    hold off;
end
