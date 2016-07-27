imgPath = 'test/';
imgType = '*.jpg'; % change based on image type
images  = dir([imgPath imgType]);
images = images(randperm(length(images)));

idx=1;
X = double(imread([imgPath images(idx).name])(:)');
y=hex2dec(images(idx).name(1));
for idx = 2:length(images)
	y=[y;hex2dec(images(idx).name(1))];
    X=[X;double(imread([imgPath images(idx).name])(:)')];
end

load Theta1.mat
load Theta2.mat
pred = predict(Theta1,Theta2,X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


