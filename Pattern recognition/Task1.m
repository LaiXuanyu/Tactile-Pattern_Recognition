
%--------------------------------------------------------------------------

% Part 1
% draw the max and min line for features and find the best single time instance
% acquire the object name and feature name, build a struct to save the data
filename_alldata = sprintf('alldata.mat');
dataStruct = load(filename_alldata);
allobjects_data = dataStruct.allobjects_data;
objects_names = fieldnames(allobjects_data);
feature_names = {'F0Electrodes', 'F0pac', 'F0pdc', 'F0tac', ...
                 'F1Electrodes', 'F1pac', 'F1pdc', 'F1tac'};
maxmin_results = struct();


for i = 1:length(objects_names)
    object_name = objects_names{i};

    maxmin_results.(object_name) = struct();
    for j = 1:length(feature_names)
        feature_name = feature_names{j};

        % Use  cal_MaxMin function to calculate the maximum and minimum values
        [max_value, min_value] = cal_MaxMin(allobjects_data.(object_name), feature_name);

        % Storing results in structures
        maxmin_results.(object_name).(sprintf('max_%s', feature_name)) = max_value;
        maxmin_results.(object_name).(sprintf('min_%s', feature_name)) = min_value;
    end
end


plot_feature(maxmin_results, objects_names, feature_names);


% draw the max and min line for features and find the best single time instance
% acquire the object name and feature name, build a struct to save the data
objects_names_two = {'KitchenSponge','CarSponge'};
maxmin_results_two = struct();


for i = 1:length(objects_names_two)
    object_name = objects_names_two{i};

    maxmin_results_two.(object_name) = struct();
    for j = 1:length(feature_names)
        feature_name = feature_names{j};

        % Use  cal_MaxMin function to calculate the maximum and minimum values
        [max_value, min_value] = cal_MaxMin(allobjects_data.(object_name), feature_name);

        % Storing results in structures
        maxmin_results_two.(object_name).(sprintf('max_%s', feature_name)) = max_value;
        maxmin_results_two.(object_name).(sprintf('min_%s', feature_name)) = min_value;
    end
end


plot_feature(maxmin_results_two, objects_names_two, feature_names);
%------------------------------------------------------------------------

% Part 2
% find the best time step that can classify different objects
% save them as F0_PVT.mat and F1_PVT.mat

% set up the time want to extract
time_step = 30; 
F0feature_names = {'F0pac', 'F0pdc', 'F0tac'};
F1feature_names = {'F1pac', 'F1pdc', 'F1tac'};

F0output_filename = 'F0_PVT.mat'; 
F1output_filename = 'F1_PVT.mat'; 

% use the function to extract particular data 
extract_and_save_data(allobjects_data, time_step, F0feature_names, F0output_filename);
extract_and_save_data(allobjects_data, time_step, F1feature_names, F1output_filename);

F0anotherfeature_names = {'F0Electrodes'};
F1anotherfeature_names = {'F1Electrodes'};

F0output_filename = 'F0_another.mat'; 
F1output_filename = 'F1_another.mat'; 

% use the function to extract particular data 
extract_and_save_data_electrodes(allobjects_data, time_step, F0anotherfeature_names, F0output_filename);
extract_and_save_data_electrodes(allobjects_data, time_step, F1anotherfeature_names, F1output_filename);


% draw F0

plot_pvt_data('F0_PVT.mat', F0feature_names);

% draw F1

plot_pvt_data('F1_PVT.mat', F1feature_names);




%-------------------------------------------------------------------------
% Function part

% find max and min from 10 statistic
function [max_feature, min_feature] = cal_MaxMin(dataArray, featurename)
    numElements = numel(dataArray);
    numPoints = 1000;

    % initilize the feature
    max_feature = -inf(1, numPoints); 
    min_feature = inf(1, numPoints);  

    for i = 1:numElements
        % ensure the length of the statistic is 1000
        featureData = dataArray(i).(featurename)(1:numPoints);

        % update Max and Min Value
        max_feature = max(max_feature, featureData);
        min_feature = min(min_feature, featureData);
    end
end

%
function plot_feature(data_struct, objects_names, feature_names)
    % Define an array of colours, one for each object.
    colors = lines(length(objects_names)); 

    % Iterative features
    for j = 1:length(feature_names)
        feature_name = feature_names{j};

        % creat a new figure for every objects
        figure;
        hold on;

        % Iterative objects
        for i = 1:length(objects_names)
            object_name = objects_names{i};

            % the length of the time
            X_axs = 1:length(data_struct.(object_name).(sprintf('max_%s', feature_name)));

            % plot same object max,min line on graph, 
            plot(X_axs, data_struct.(object_name).(sprintf('max_%s', feature_name)), 'Color', colors(i,:), 'DisplayName', [object_name ' max']);
            plot(X_axs, data_struct.(object_name).(sprintf('min_%s', feature_name)), 'Color', colors(i,:), 'LineStyle', '--', 'DisplayName', [object_name ' min']);
        end

        % set the title
        title(sprintf('Feature: %s', feature_name));
        legend('show');
        xlabel('Time');
        ylabel('Value');

        hold off;
    end
end

function extract_and_save_data(originalData, time_step, feature_names, output_filename)
    newData = struct();

    % find all of the object
    object_names = fieldnames(originalData);

    % find all of the object
    for i = 1:length(object_names)
        object_name = object_names{i};

        newData.(object_name) = struct();

        % Iterative each trials
        for trial = 1:10
            % initilize a struct to save value
            trialData = struct();

            for j = 1:length(feature_names)
                feature_name = feature_names{j};

                % acquire the data through given time instance
                trialData.(feature_name) = originalData.(object_name)(trial).(feature_name)(time_step);
            end

            % save the data into struct
            newData.(object_name).(sprintf('Trial%d', trial)) = trialData;
        end
    end

    % save the data into certain type
    save(output_filename, 'newData');
end

function plot_pvt_data(file, feature_names)
    % load the datq
    data = load(file);

    % give each objects a paticular color
    object_names = fieldnames(data.newData);
    colors = lines(length(object_names));
    newFilename = strrep(file, '_', '-');
    % create 3D scatter graph
    figure;
    hold on;

    for i = 1:length(object_names)
        object_name = object_names{i};

        % extra the data, the function is below this funtion
        [feature1, feature2, feature3] = extract_pvt_data(data.newData.(object_name), feature_names);

        % draw the data
        scatter3(feature1, feature2, feature3, 'DisplayName', object_name , 'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', colors(i,:), 'SizeData', 50);
    end
    
    % set the graph
    xlabel(feature_names{1});
    ylabel(feature_names{2});
    zlabel(feature_names{3});
    title([newFilename ' Data']);
    view(3)
    legend('show');
    grid on
    hold off;
end

function [feature1, feature2, feature3] = extract_pvt_data(data_struct, feature_names)
    feature1 = [];
    feature2 = [];
    feature3 = [];
    trialNames = fieldnames(data_struct);
    
    % Iterative each trials
    for i = 1:length(trialNames)
        trialName = trialNames{i};
        trialData = data_struct.(trialName);
        
        % extract and save data
        feature1 = [feature1; trialData.(feature_names{1})];
        feature2 = [feature2; trialData.(feature_names{2})];
        feature3 = [feature3; trialData.(feature_names{3})];
    end
end

function extract_and_save_data_electrodes(originalData, time_step, feature_names, output_filename)
    % Initialize the main cell array to store the extracted data
    extractedData = struct();

    % Iterate over all objects in the originalData
    object_names = fieldnames(originalData);
    for i = 1:length(object_names)
        object_name = object_names{i};

        % Initialize a cell array to store data for this object
        extractedData.(object_name) = struct();

        % Iterate over each trial for the current object
        for trial = 1:10
            % Initialize a struct to save values for this trial
            trialData = struct();

            % Iterate over each feature name
            for j = 1:length(feature_names)
                feature_name = feature_names{j};

                % Extract the data at the given time instance and save only that
                trialData.(feature_name) = originalData.(object_name)(trial).(feature_name)(:, time_step);
            end

            % Save the trial data into the object data array
            objectData.(sprintf('Trial%d', trial))  = trialData;
        end

        % Store the object data in the main extracted data array
        extractedData.(object_name)= objectData;
    end

    % Save the extracted data to a file
    save(output_filename, 'extractedData');
end


