trainPercent = 60;
crossValidatePercent = 80;
testPercent = 100;

try
  load X.mat;
  load y.mat;
catch
  saveImages();
end
m = size(X,1);
input_layer_size = size(X,2);
hidden_layer_size = ceil(sqrt(input_layer_size));
num_labels = length(unique(y));

trainNum = ceil(size(X,1)*trainPercent/100);
cvNum = ceil(size(X,1)*crossValidatePercent/100);
trainX = X(1:trainNum,:);
trainy = y(1:trainNum,:);
cvX = X(trainNum:cvNum,:);
cvy = y(trainNum:cvNum,:);
testX = X(cvNum:end,:);
testy = y(cvNum:end,:);

errors = zeros(m,2);
try
  load NoFileFound.mat;
	% load Theta1.mat
	% load Theta2.mat
catch
	Theta1 = randomInitWeight(input_layer_size, hidden_layer_size);
	Theta2 = randomInitWeight(hidden_layer_size, num_labels);
end

unrolledTheta = [Theta1(:) ; Theta2(:)];


% Train
lambda = 10;

% for iter = 1:trainNum
  % trainX = X(1:iter,:);
  % trainy = y(1:iter,:);
  options = optimset('MaxIter', 200);

costF = @(p) costFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, trainX, trainy, lambda);

% Create "short hand" for the cost function to be minimized
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[unrolledTheta, cost] = fmincg(costF, unrolledTheta, options);

% Obtain Theta1 and Theta2 back from unrolledTheta
Theta1 = reshape(unrolledTheta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(unrolledTheta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
  pred = predict(Theta1,Theta2,trainX);
  % errors(iter,1) = sum(double(pred - trainy).^2);

save Theta1.mat Theta1
save Theta2.mat Theta2

  % pred = predict(Theta1,Theta2,cvX);
  % errors(iter,2) = sum(double(pred - cvy).^2);

% end

% save errors.mat errors

pred = predict(Theta1,Theta2,testX);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == testy)) * 100);


