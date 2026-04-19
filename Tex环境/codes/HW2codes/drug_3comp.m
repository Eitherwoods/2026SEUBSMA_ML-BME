function dxdt = drug_3comp(~, x, k21, k23, k32, k02, k03, f10)
% 状态变量
x1 = x(1); x2 = x(2); x3 = x(3);
% 微分方程
dx1 = f10 - k21*x1;
dx2 = k23*x3 + k21*x1 - k32*x2 - k02*x2;
dx3 = k32*x2 - k23*x3 - k03*x3;
dxdt = [dx1; dx2; dx3];
end