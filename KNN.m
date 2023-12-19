function indexes = KNN(k,x,data,U,l)

[~,n]=size(data);
x_bar=mean(data,2);
w_x=compute_w(x,U,x_bar,l);

val=inf*ones(1,k);
indexes=-1*ones(1,k);

for i = 1:n
    w_i=compute_w(data(:,i),U,x_bar,l);
    dist=norm(w_x-w_i,2);
    [m,pos]=max(val);
    if dist<m
        val(pos)=dist;
        indexes(pos)=i;
    end
end