% 美国人口数据 (千人)
year_data = (1790:10:1880)';
pop_data = [3929; 5308; 7240; 9638; 12866; 17069; 23192; 31443; 38558; 50156];
t_data = year_data - year_data(1);   % 以1790年为t=0
n = length(t_data);

% --- Malthusian 模型：线性最小二乘 ---
y_m = log(pop_data / pop_data(1));
r_malthus = sum(t_data .* y_m) / sum(t_data.^2);
fprintf('Malthusian r = %.5f\n', r_malthus);

% --- Logistic 模型：多组等时间间隔三点法 ---
% 收集所有可能的等间隔三点组合
K_estimates = [];
for i = 1:n
    for j = i+1:n
        k = 2*j - i;       % 使得三点等间隔: t_i, t_j, t_k
        if k <= n && k > j
            A = pop_data(i); B = pop_data(j); D = pop_data(k);
            denom = A*D - B^2;
            if abs(denom) > 1e-12
                K = (B*(2*A*D - B*(A+D))) / denom;
                if K > 0 && isfinite(K)
                    K_estimates = [K_estimates; K];
                end
            end
        end
    end
end
if ~isempty(K_estimates)
    K_logistic = median(K_estimates);   % 取中位数
    fprintf('Logistic 容纳量 K (中位数) = %.1f\n', K_logistic);
    fprintf('有效三点组合数 = %d\n', length(K_estimates));
else
    error('无有效三点组合');
end

% 对全部数据线性回归估计 r
y_l = log(pop_data ./ (K_logistic - pop_data));
X = [ones(n,1), t_data];
b = X \ y_l;   % b(1)=C, b(2)=r
r_logistic = b(2);
C_logistic = b(1);
fprintf('Logistic r (线性回归) = %.5f\n', r_logistic);

% --- 预测 ---
year_pred = (1890:10:1980)';
pop_true = [62948; 75995; 91972; 105711; 122775; 131669; 150697; 179323; 203185; 226500];
t_pred = year_pred - year_data(1);

% Malthusian 预测
N_pred_malthus = pop_data(1) * exp(r_malthus * t_pred);

% Logistic 预测
N0 = pop_data(1);
A_val = K_logistic/N0 - 1;
N_pred_logistic = K_logistic ./ (1 + A_val * exp(-r_logistic * t_pred));

% 相对误差
err_m = abs(N_pred_malthus - pop_true) ./ pop_true * 100;
err_l = abs(N_pred_logistic - pop_true) ./ pop_true * 100;

% 输出表格
fprintf('\n年份   实际     Malthusian Logistic   Malthus误差 Logistic误差\n');
for i = 1:length(year_pred)
    fprintf('%d  %6d  %8.0f  %8.0f  %8.2f%%  %8.2f%%\n', ...
        year_pred(i), pop_true(i), N_pred_malthus(i), N_pred_logistic(i), ...
        err_m(i), err_l(i));
end

% --- 绘图 ---
figure;
t_plot = linspace(0, 200, 300)';
plot(year_data, pop_data, 'ko', 'MarkerFaceColor', 'k'); hold on;
plot(year_pred, pop_true, 'k^', 'MarkerFaceColor', 'k');
plot(year_data(1)+t_plot, pop_data(1)*exp(r_malthus*t_plot), 'b--', 'LineWidth',1.5);
plot(year_data(1)+t_plot, K_logistic./(1 + (K_logistic/pop_data(1)-1)*exp(-r_logistic*t_plot)), 'r-', 'LineWidth',1.5);
xlabel('年份'); ylabel('人口 (千人)');
legend('建模数据 (1790-1880)', '实际值 (1890-1980)', ...
       'Malthusian预测', 'Logistic预测', 'Location', 'northwest');
title('美国人口预测: Malthusian vs Logistic (多组三点法)');
grid on;