function [gt,arranged_traj] = arrange_cam_policy(gt,trajectory,db_no)

num_cam = unique(trajectory(:,1));

arranged_traj = [];
for c = 1:length(num_cam)
    traj_this_c = trajectory( trajectory(:,1)==c, :);
    % sort time index
    [~,tindex] = sort(traj_this_c(:,2));
    traj_this_c = traj_this_c(tindex,:);
    arranged_traj = [arranged_traj; traj_this_c];
end

% take only test people for GT
p_1 = [88 137 182 223 174 71 200 58 1 185 74 168 235 46 17 67 162 100, ...
    192 113 140 6 94 81 11 181 159 78 147 22 127 115 68 143 59 212 217 161,...
    92 164 227 206 73 123 125 126 230 233 83 197 231 34 145 26 114 27 138 7, ...
    151 8 48 120 166 215 35 85 64  111 156 225 65 171 153 25 45 207 219 178,...
    23 165 50 199 110 203 211 47 152 30 132 102 205 96 89 129 52 107 60 36,...
    216 167 87 149 208 116 213 57 41 214 12 175 163 15 173 144 134 86 194 82,...
    128 186 63 105 122 69 21 183 169 187 222 19 232 108 198 79 141 91 51 150,...
    53 77 5 119 39 33 170 84 229 180 133 40 188 139 54 121 158 55 42 10 9 124,...
    142 90 136 189 226 131 135 13 95 24 209 191 44 29 16 218 20 93 184 130 117,...
    49   204     3   112   146   195   109    62    37    98   154    72    99,...
    157  234   103    56   106    31    66   172   202 38 43 28 101 221 193 177,...
    2 4 160 155 14 210 196 176 97 70 201 228 75 18 104 179 118 148 190 224 220,...
    32 61 80 76];
p_3 = [9     1     5     2     8     7     6    12    13    11     4    14    10     3];
p_4 = [1 46 17 6 11 22 34 26 27 7 8 48 35 25 45 23 47 30 36 41 12 15 21 19  5 39 33 40 42 10 9 13 24 44 29 16 20 49 3 37 31 38 43 28 2 4 14 18 32];

% p contains train person
if db_no == 1
    p = p_1(1:2:end);
elseif db_no == 3
    p = p_3(1:2:end);
elseif db_no == 4
    p = p_4(1:2:end);
end

if db_no ~= 2
    for i = 1:length(p)
        gt(gt(:,3)==p(i)-1,:) = [];
    end
end

% convert trajectory indices
arranged_traj(:,[2,3]) = arranged_traj(:,[2,3])-1;
