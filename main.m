%%
clear
clc
close all

%% Data extraction

% Training set
adr = './database/training1/';
[data_trn,P,N,Nc,size_cls_trn] = loaddatabase(adr);

% %Display the database
displaydatabase(data_trn,Nc,size_cls_trn)

%% Parameters

[h,w,c]=size(imread("database\test1\yaleB01_P00A+005E+10.pgm"));
alpha=0.9;
k=5;

%% Dataprocessing

x_bar=mean(data_trn,2);
data_centered=data_trn-x_bar;
X=data_centered/sqrt(N);
Gram=X'*X;
[V,D]=eig(Gram);
[eiglist,indexes]=sort(diag(D),"descend");
eiglist = eiglist(1:N-1);
indexes=indexes(1:N-1)-1;
V = V(:, indexes);
U = X*V*((V'*Gram*V)^(-1/2));

%%% Computing l
kappaList=zeros(1,N-5);
for j=1:N-1
    kappaList(j)=sum(eiglist(1:j))/sum(eiglist);
end

l = sum(kappaList<alpha);
% l=5;

%%%

%%% eigenfaces display
figure
for i = 1:Nc
    if (i==Nc)
        for j = 1:N/Nc-1
            k = (i-1)*(N/Nc) + j;
            subplot(Nc, N/Nc, k);
            imagesc(reshape(U(:,k), [h, w]));
            colormap('gray')
        end
    else
        for j = 1:N/Nc
            k = (i-1)*(N/Nc) + j;
            subplot(Nc, N/Nc, k);
            imagesc(reshape(U(:,k), [h, w]));
            colormap('gray')
        end
    end
end
% %%% 
% 
% %%% reconstructed face display

x1_centered=data_trn(:,1)-x_bar;
tmp1=x1_centered'*U(:,1:l);
out=zeros(length(x1_centered),l);
for i = 1:l 
    out(:,i)=tmp1(i)*U(:,i);
end
x1estim=sum(out,2)+x_bar;
figure, 
subplot 121
imagesc(reshape(x1estim,[192,168])),colormap('gray')
subplot 122
imagesc(reshape(data_trn(:,1),[192,168])),colormap('gray')

%%% Classification

% Testing set
adr = './database/test1/';
[data_tes,P2,N_test,Nc2,size_cls_tes] = loaddatabase(adr);

displaydatabase(data_tes,Nc2,size_cls_tes)

%%% KNN

% est_lb=compute_est_lb_withKNN(data_tes,data_trn,k,U,l,N_test);
% 
% true_lb=[1*ones(1,size_cls_tes(1)) , 2*ones(1,size_cls_tes(2)) , 3*ones(1,size_cls_tes(3)) , 4*ones(1,size_cls_tes(4)) , 5*ones(1,size_cls_tes(5)) , 6*ones(1,size_cls_tes(6))];
% 
% C = confusionmat(true_lb',est_lb');
% 
% figure,imagesc(C)

%%% Gaussian

est_lb=GAUSS(data_tes,data_trn,U,l,Nc2,size_cls_trn);

true_lb=[1*ones(1,size_cls_tes(1)) , 2*ones(1,size_cls_tes(2)) , 3*ones(1,size_cls_tes(3)) , 4*ones(1,size_cls_tes(4)) , 5*ones(1,size_cls_tes(5)) , 6*ones(1,size_cls_tes(6))];

C = confusionmat(true_lb',est_lb');

figure,imagesc(C)