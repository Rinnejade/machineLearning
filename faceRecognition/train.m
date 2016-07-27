imgPath = 'face/';
imgType = '*.jpg'; % change based on image type
images  = dir([imgPath imgType]);
images = images(randperm(length(images)));

num_labels=2;
idx=1;
Xtrain = double(imread([imgPath images(idx).name])(:)');
ytrain=hex2dec(images(idx).name(1));
for idx = 2:length(images)
	ytrain=[ytrain;hex2dec(images(idx).name(1))];
    Xtrain=[Xtrain;double(imread([imgPath images(idx).name])(:)')];
end

imgPath = 'test/';
imgType = '*.jpg'; % change based on image type
images  = dir([imgPath imgType]);
images = images(randperm(length(images)));

num_labels=2;
idx=1;
Xtest = double(imread([imgPath images(idx).name])(:)');
ytest=hex2dec(images(idx).name(1));
for idx = 2:length(images)
	ytest=[ytest;hex2dec(images(idx).name(1))];
    Xtest=[Xtest;double(imread([imgPath images(idx).name])(:)')];
end

input_layer_size = size(X,2);
hidden_layer_size = ceil(sqrt(input_layer_size));

try
	load Theta1.mat
	load Theta2.mat
catch
	Theta1 = randomInitWeight(input_layer_size, hidden_layer_size);
	Theta2 = randomInitWeight(hidden_layer_size, num_labels);
end

unrolledTheta = [Theta1(:) ; Theta2(:)];

lambda = 1;
iterations = 1;
options = optimset('MaxIter', 200);

costF = @(p) costFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xtrain, ytrain, lambda);

lambda = 1;

% Create "short hand" for the cost function to be minimized

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[unrolledTheta, cost] = fmincg(costF, unrolledTheta, options);

% Obtain Theta1 and Theta2 back from unrolledTheta
Theta1 = reshape(unrolledTheta(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(unrolledTheta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save Theta1.mat Theta1
save Theta2.mat Theta2

pred = predict(Theta1,Theta2,Xtest);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);


