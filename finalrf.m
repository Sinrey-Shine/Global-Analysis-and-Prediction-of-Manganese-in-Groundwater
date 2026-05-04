
clear; clc;


rng(123);  % 保证可复现


T = readtable("C:\Users\admin\Desktop\第30_ExportTable.csv");


X = T{:, 5:34};   
Y = double(T{:, 4});     


cvHold = cvpartition(Y,'HoldOut',0.2,'Stratify',true); % 80%训练，20%测试
idxTrain = training(cvHold);
idxTest  = test(cvHold);

XTrain = X(idxTrain,:);
YTrain = Y(idxTrain);
XTest  = X(idxTest,:);
YTest  = Y(idxTest);



 optVars = [
    optimizableVariable('NumTrees', [50, 1000], 'Type', 'integer')
     optimizableVariable('MinLeafSize', [1, 50], 'Type', 'integer')
     optimizableVariable('NumVariablesToSample', [1, size(X,2)], 'Type', 'integer')
 ];
 results = bayesopt(@(params) rfObjFcn(params, XTrain, YTrain), ...
     optVars, ...
     'AcquisitionFunctionName', 'expected-improvement-plus', ...
     'MaxObjectiveEvaluations', 30, ...
     'Verbose', 1, ...
     'UseParallel', true);
 bestParams = bestPoint(results);


rng(123);
finalModel = TreeBagger( ...
    bestParams.NumTrees, XTrain, YTrain, ...
    'Method', 'classification', ...
    'MinLeafSize', bestParams.MinLeafSize, ...
    'NumVariablesToSample', bestParams.NumVariablesToSample, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on', ... 
    'Options', statset('UseParallel', true));

save('RFModel4545645646.mat', 'finalModel', 'bestParams');



thresh = 0.38; 

[~, scores] = predict(finalModel, XTest);
Ypred = double(scores(:,2) >= thresh);
Ytrue = double(YTest);


confMatrix = confusionmat(Ytrue, Ypred);


accuracy    = sum(Ypred == Ytrue) / length(Ytrue);
sensitivity = confMatrix(2,2) / (confMatrix(2,2) + confMatrix(2,1));
specificity = confMatrix(1,1) / (confMatrix(1,1) + confMatrix(1,2));
precision   = confMatrix(2,2) / (confMatrix(2,2) + confMatrix(1,2));
F1          = 2 * (precision * sensitivity) / (precision + sensitivity);


[~,~,~,AUC] = perfcurve(Ytrue, scores(:,2), 1);


Po = accuracy;
Pe = ((sum(Ytrue==0)/length(Ytrue))*(sum(Ypred==0)/length(Ypred))) + ...
     ((sum(Ytrue==1)/length(Ytrue))*(sum(Ypred==1)/length(Ypred)));
kappa = (Po - Pe) / (1 - Pe);


fprintf('\n===== 测试集评估结果 (阈值=%.2f) =====\n', thresh);
fprintf('Accuracy:      %.2f%%\n', accuracy * 100);
fprintf('Sensitivity:   %.2f%%\n', sensitivity * 100);
fprintf('Specificity:   %.2f%%\n', specificity * 100);
fprintf('Precision:     %.2f%%\n', precision * 100);
fprintf('F1 Score:      %.2f\n', F1);
fprintf('AUC:           %.3f\n', AUC);
fprintf('Cohen''s Kappa: %.3f\n', kappa);


[X_roc, Y_roc, ~, ~] = perfcurve(Ytrue, scores(:,2), 1);
figure;
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
xlabel('假阳性率 (FPR)');
ylabel('真正率 (TPR)');
title(['测试集 ROC 曲线 (AUC = ' num2str(AUC,'%.3f') ')']);
grid on;


thresholds = 0.1:0.0001:0.9;
sens_vals = zeros(size(thresholds));
spec_vals = zeros(size(thresholds));

for i = 1:length(thresholds)
    t = thresholds(i);
    Ypred_t = double(scores(:,2) >= t);
    cm = confusionmat(Ytrue, Ypred_t);
    if size(cm,1) == 1
        cm(2,2)=0; cm(2,1)=0;
    end
    TP = cm(2,2); FP = cm(1,2);
    FN = cm(2,1); TN = cm(1,1);
    sens_vals(i) = TP / (TP + FN + eps);
    spec_vals(i) = TN / (TN + FP + eps);
end

figure;
plot(thresholds, sens_vals, '-r', 'LineWidth', 2); hold on;
plot(thresholds, spec_vals, '-b', 'LineWidth', 2);
xlabel('阈值 (Threshold)');
ylabel('值');
legend('Sensitivity','Specificity','Location','best');
title('Sensitivity 与 Specificity 随阈值变化');
grid on;

[~, idx_bal] = min(abs(sens_vals - spec_vals));
plot(thresholds(idx_bal), sens_vals(idx_bal), 'ko', 'MarkerSize',8,'MarkerFaceColor','k');
text(thresholds(idx_bal), sens_vals(idx_bal), sprintf('  Threshold=%.2f', thresholds(idx_bal)),'FontSize',10);



%% ==================== OOB Feature Importance ====================
imp = finalModel.OOBPermutedPredictorDeltaError;  % Get importance of each feature

% Feature labels F1~F30
numFeatures = length(imp);
featureLabels = arrayfun(@(x) ['F' num2str(x)], 1:numFeatures, 'UniformOutput', false);

% Plot
figure;
bar(imp);
xlabel('Feature');
ylabel('OOB Importance');
title('OOB Feature Importance (Original Order)');
set(gca, 'XTick', 1:numFeatures, 'XTickLabel', featureLabels, 'XTickLabelRotation', 45);
grid on;


%% ==================== 贝叶斯优化目标函数 ====================
function objective = rfObjFcn(params, X, Y)
    rng(123);
    cv = cvpartition(Y, 'KFold', 5, 'Stratify', true);
    auc_list = zeros(cv.NumTestSets,1);
    for k = 1:cv.NumTestSets
        idxTrain = training(cv, k);
        idxTest  = test(cv, k);
        XTrain = X(idxTrain,:); YTrain = Y(idxTrain);
        XTest  = X(idxTest,:);  YTest  = Y(idxTest);
        model = TreeBagger( ...
            params.NumTrees, XTrain, YTrain, ...
            'Method','classification', ...
            'MinLeafSize', params.MinLeafSize, ...
            'NumVariablesToSample', params.NumVariablesToSample, ...
            'OOBPrediction','off', ...
            'Options', statset('UseParallel', true));
        [~, scores] = predict(model, XTest);
        [~,~,~,auc] = perfcurve(YTest, scores(:,2), 1);
        auc_list(k) = auc;
    end
    objective = 1 - mean(auc_list);  % 最大化AUC
end


%% ==================== Threshold scan (NC-style) ====================

thresholds = 0.0:0.001:1.0;

sens_vals = zeros(size(thresholds));
spec_vals = zeros(size(thresholds));
acc_vals  = zeros(size(thresholds));

for i = 1:length(thresholds)
    t = thresholds(i);
    Ypred_t = double(scores(:,2) >= t);

    cm = confusionmat(Ytrue, Ypred_t);

    % 防止只出现一类的情况
    if size(cm,1) == 1
        if unique(Ytrue) == 0
            cm = [cm(1,1), 0; 0, 0];
        else
            cm = [0, 0; 0, cm(1,1)];
        end
    end

    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);

    sens_vals(i) = TP / (TP + FN + eps);
    spec_vals(i) = TN / (TN + FP + eps);
    acc_vals(i)  = (TP + TN) / (TP + TN + FP + FN + eps);
end

%% ===== 找最优阈值（accuracy 最大）=====
[~, idx_opt] = max(acc_vals);
opt_thresh = 0.38;

%% ==================== Plot ====================
figure('Color','w'); hold on;

plot(thresholds, sens_vals, 'r-', 'LineWidth', 2);
plot(thresholds, acc_vals,  'b-', 'LineWidth', 2);
plot(thresholds, spec_vals, 'g-', 'LineWidth', 2);

% 竖直虚线（最优 cutoff）
xline(opt_thresh, 'k--', 'LineWidth', 1.5);

%% 坐标 & 样式
xlim([0 1]);
ylim([0 1]);

xlabel('cutoff', 'FontSize', 12);
ylabel('classification proportion', 'FontSize', 12);

lgd = legend({'sensitivity','accuracy','specificity'}, ...
             'Location','southwest', 'Box','off');

lgd.ItemTokenSize = [10, 10];   % ⭐ 让图例线段变短 → 视觉上往左
lgd.Units = 'normalized';       % 用归一化坐标（关键）
lgd.Position(1) = lgd.Position(1) + 0.05;  % ⭐ 整体向右移动

set(gca, ...
    'FontSize', 11, ...
    'LineWidth', 1.2, ...
    'TickDir','out');

grid off;
box on;



%% ===== 找最优阈值（accuracy 最大）=====
[~, idx_opt] = max(acc_vals);
opt_thresh = 0.38;

%% ==================== Plot ====================
figure('Color','w'); hold on;

plot(thresholds, sens_vals, 'r-', 'LineWidth', 2.5);
plot(thresholds, acc_vals,  'b-', 'LineWidth', 2.5);
plot(thresholds, spec_vals, 'g-', 'LineWidth', 2.5);

% 竖直虚线（最优 cutoff）
xline(opt_thresh, 'k--', 'LineWidth', 2);

%% 坐标 & 样式
xlim([0 1]);
ylim([0 1]);

xlabel('Cutoff', 'FontSize', 30);
ylabel('Classification proportion', 'FontSize', 30);

lgd = legend({'Sensitivity','Accuracy','Specificity'}, ...
             'Location','southwest', ...
             'Box','off', ...
             'FontSize',30);   % ⭐ legend字体

lgd.ItemTokenSize = [15, 12];   % legend线段大小
lgd.Units = 'normalized';
lgd.Position(1) = lgd.Position(1) + 0.05;

set(gca, ...
    'FontSize', 30, ...        % ⭐ 坐标轴数字大小（关键）
    'LineWidth', 1.5, ...
    'TickDir','out', ...
    'FontName','Arial');       % ⭐ 论文常用字体

grid off;
box on;