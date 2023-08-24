#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/select.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <sqlite3.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <time.h>

#define AVATAR_LEN 3  // 图标的数量
#define ID_LEN 6  // 用户id长度
#define SAFE_LEN 8 // 独立服务密码长度
#define NAME_LEN 32 // 用户名长度
#define PW_LEN 32 // 密码长度
#define TIME_LEN 64 // 时间戳长度
#define CONTENT_LEN 1034 // 消息内容长度


//数据库
sqlite3 *db = NULL;
//全局变量，每次使用前都会先清空
int cnt_users = 0, res, cnt_msgs = 0;
char feedback[1024];


/*数出用户总数的回调函数*/
int checkTable_Callback(void *unnecessary_here, int num_columns_in_result_row, char **value_of_count, 
        char **label_for_count) { /* will be COUNT(*) normally, but modified in this case*/
    int i = 0;
    for(; i < strlen(value_of_count[0]); i++){
        cnt_users = cnt_users * 10 + (value_of_count[0][i] - '0');
    }
    // printf("initial count = %d\n",cnt);
    return 0;
}

/*数出未读消息总数的回调函数*/
int checkMsg_Callback(void *unnecessary_here, int num_columns_in_result_row, char **value_of_count, 
        char **label_for_count) { 
    int i = 0;
    cnt_msgs = 0;
    for(; i < strlen(value_of_count[0]); i++){
        cnt_msgs = cnt_msgs * 10 + (value_of_count[0][i] - '0');
    }
    return 0;
}


/*检查目标用户和密码是否存在的回调函数
在登录、发送消息检查目标用户是否在线、修改密码时调用
value_of_count在这里只可能取0或1*/
int checkUser_Callback(void *unnecessary_here, int num_of_columns_in_result_row, char **value_of_count, 
        char **label_for_count) { 
    int i = 0;
    res = 0;
    for(; i < strlen(value_of_count[0]); i++){
        res = res * 10 + (value_of_count[0][i] - '0');
    }
    return 0;
}


/*将用户设置为在线状态*/
int setOnline(char *id, int fd){
    /*将当前用户id和这个连接的fd插入数据库*/
    char setOnline_sql[256] = "INSERT INTO Online(id,fd) VALUES('";
    char fd_str[10], *sqlError;
    //itoa(fd,fd_str,10);
    //int转化为字符串
    snprintf(fd_str, sizeof(fd_str), "%d", fd);
    strcat(setOnline_sql,id); 
    strcat(setOnline_sql,"',");
    strcat(setOnline_sql,fd_str);
    strcat(setOnline_sql,");");
    // printf("setOnline sql = %s\n",setOnline_sql);
    return sqlite3_exec(db, setOnline_sql, NULL, NULL, &sqlError);
}


/*根据id查询当前在线用户的fd
  由于多次使用而封装成独立函数*/
int findReceiverFd(char *receiver_id){
    int receiver_fd = 0, num_rows, num_cols, queryres;
    char **query_resultset = NULL;
    char *errmsg = NULL;
    char searchReceiver_sql[64] = "SELECT fd FROM Online WHERE id='";
    strcat(searchReceiver_sql,receiver_id);
    strcat(searchReceiver_sql,"';");
    queryres = sqlite3_get_table(db, searchReceiver_sql, &query_resultset, &num_rows, &num_cols, &errmsg);
    if(queryres != 0){
        printf("get_table error : %s\n",errmsg);
        return;
    }else{
        if(num_rows != 1)
            printf("duplicate users!\n");
        else{   //从结果表中取出并从字符串转化为int
            int k;
            for(k = 0; k < strlen(query_resultset[1]); k++)
                receiver_fd = receiver_fd * 10 + (query_resultset[1][k] - '0');
                // printf("receiver fd = %d\n",receiver_fd);
            }
    }
    sqlite3_free_table(query_resultset);
    return receiver_fd;
}


/*获取这两个用户的聊天记录*/
void getChatHistory(char *id1, char* id2, int fd){
    char sql[1050] = {0};
    strcpy(sql, "SELECT msg FROM ChatHistory WHERE (sender='");
    strcat(sql, id1);
    strcat(sql, "' AND receiver='");
    strcat(sql, id2);
    strcat(sql, "') OR (sender='");
    strcat(sql, id2);
    strcat(sql, "' AND receiver='");
    strcat(sql, id1);
    strcat(sql, "') ORDER BY msg ASC;");
    // printf("get chat history: %s\n", sql);

    char **query_resultset = NULL;
    char *errmsg = NULL;
    int num_rows = 0, num_cols = 0, i;

    if(sqlite3_get_table(db, sql, &query_resultset, &num_rows, &num_cols, &errmsg) != 0){
        printf("get chat history error: %s\n", errmsg);
    }else{
        for(i = 1; i < num_rows + 1; i++){  //这里num_cols = 1
            send(fd, query_resultset[i], strlen(query_resultset[i]), 0);  //发送用户信息
            //printf("query_resultset :%s\n",query_resultset[i]);
            usleep(20000);
        }
        sqlite3_free_table(query_resultset);
    }

    return;
}


/*4-2*/
void getMsgList(char *user_id, int fd, char **friendslist, int num_friends){
    //反馈给客户端：#8|好友id|未读消息数|最新消息&
    char *errmsg_1 = NULL;
    char *errmsg_2 = NULL;
    int num_rows = 0, num_cols = 0, i;
    char *cur_friend_id = NULL;
    char **content_set  = NULL;
    char num_str[5] = {0};
    // printf("%d\n", num_friends);

    //查找最新消息和未读消息数量
    for(i = 1; i < num_friends + 1; i++){   //这里num_cols = 1
        cur_friend_id = friendslist[i];
        char latestmsg_sql[512] = "SELECT msg FROM ChatHistory WHERE (sender='";
        strcat(latestmsg_sql, user_id);
        strcat(latestmsg_sql, "' AND receiver='");
        strcat(latestmsg_sql, cur_friend_id);
        strcat(latestmsg_sql, "') OR (sender='");
        strcat(latestmsg_sql, cur_friend_id);
        strcat(latestmsg_sql, "' AND receiver='");
        strcat(latestmsg_sql, user_id);
        strcat(latestmsg_sql, "') ORDER BY msg DESC LIMIT 1;");
        // printf("latest msg: %s\n", latestmsg_sql);

        char numUnread_sql[256] = "SELECT COUNT(*) FROM ChatHistory WHERE sender='";
        strcat(numUnread_sql, cur_friend_id);
        strcat(numUnread_sql, "' AND receiver='");
        strcat(numUnread_sql, user_id);
        strcat(numUnread_sql, "' AND read=0;");
        // printf("num_sql: %s\n", numUnread_sql);

        if(sqlite3_exec(db, numUnread_sql, checkMsg_Callback, 0, &errmsg_1) != 0 || sqlite3_get_table(db, latestmsg_sql, &content_set, &num_rows, &num_cols, &errmsg_2) != 0){
            printf("error_1: %s\n",errmsg_1);
            printf("error_2: %s\n",errmsg_2);
        }else{
            if(num_rows == 0)
                continue;
            sprintf(num_str, "%d", cnt_msgs);
            // printf("num = %s\n", num_str);
            strcpy(feedback, "#8|");
            strcat(feedback, cur_friend_id);
            strcat(feedback, "|");
            strcat(feedback, num_str);
            strcat(feedback, "|");
            strcat(feedback, content_set[1]);
            strcat(feedback, "&");
            // printf("feedback:%s\n",feedback);
            send(fd, feedback, strlen(feedback), 0);
        }         
    }
    // printf("before return.\n");
    return;
}


/*4-1 获取好友列表*/
void getFriendsList(char *data, int fd){
    //客户端请求：#4|id;
    //反馈给客户端：#4|id|username|avatar&
    
    int type = 0;
    char user_id[ID_LEN] = {0};

    //提取用户登录信息
    sscanf(data, "#%d|%[^&]", &type, user_id);

    //首先获取好友列表
    char friendsList_sql[512] = "SELECT * FROM (SELECT id1 AS id FROM FriendRelation WHERE FriendRelation.id2='";
    strcat(friendsList_sql, user_id);
    strcat(friendsList_sql, "' UNION ALL SELECT id2 AS id FROM FriendRelation WHERE FriendRelation.id1='");
    strcat(friendsList_sql, user_id);
    strcat(friendsList_sql, "');");
    // printf("friendslist: %s\n",friendsList_sql);

    char **friendslist = NULL;
    char *errmsg = NULL;
    int num_friends, num_attr, i, j;

    if(sqlite3_get_table(db, friendsList_sql, &friendslist, &num_friends, &num_attr, &errmsg) != 0){
        printf("get friends error: %s\n",errmsg);
    }else{
        char **friendsInfo_list = NULL;
        int num_rows = 0, num_cols = 0;
        char friendsInfo_sql[512] = "SELECT id, username, avatar FROM UserInfo WHERE id IN (SELECT id1 AS id FROM FriendRelation WHERE FriendRelation.id2='";
        strcat(friendsInfo_sql, user_id);
        strcat(friendsInfo_sql, "' UNION ALL SELECT id2 AS id FROM FriendRelation WHERE FriendRelation.id1='");
        strcat(friendsInfo_sql, user_id);
        strcat(friendsInfo_sql, "');");
        // printf("%s\n",friendsInfo_sql);
        if(sqlite3_get_table(db, friendsInfo_sql, &friendsInfo_list, &num_rows, &num_cols, &errmsg)){
            printf("get attr error: %s\n",errmsg);
        } else{
            for(i = 1; i < num_rows + 1; i++){   //选择每一个好友的信息
                strcpy(feedback, "#4");
                for(j = 0; j < num_cols; j++){
                    strcat(feedback, "|");
                    strcat(feedback, friendsInfo_list[i * num_cols + j]);
                }
                strcat(feedback, "&");
                send(fd, feedback, strlen(feedback), 0);
                usleep(200000);   //防止客户端接收出错
            }
            usleep(40000);
            // printf("before msg\n");
            getMsgList(user_id,fd,friendslist,num_friends);  //获取消息列表和未读数目
        }
    }

    return;
} 


/*2 处理登录请求*/
void logIn(char *data, int fd){
    //客户端传来：#2|id|password&
    int type = 0;
    char id[ID_LEN] = {0}, pw[PW_LEN] = {0};

    //提取用户登录信息
    sscanf(data, "#%d|%[^|]|%[^&]", &type, id, pw);

    //sql查询命令，检验该用户是否存在
    char sql[128] = "SELECT COUNT(*) FROM UserInfo WHERE id='";
    strcat(sql,id);
    strcat(sql,"' AND password='");
    strcat(sql,pw);
    strcat(sql,"';");
    // printf("checkuser sql = %s\n",sql);

    //查询数据库，服务器回传：#2|结果|反馈信息&
    char *sqlError = NULL;
    if (sqlite3_exec(db, sql, checkUser_Callback, 0, &sqlError) != 0){
        send(fd, "#2|0|system error&", strlen("#2|0|system error&"), 0);
        printf("login error = %s\n",sqlError);
    } else{
        if(res == 1){  //通过身份验证
            //如果登陆成功，加入在线表，返回用户名和头像编号
            if(setOnline(id, fd) == 0){
                strcpy(feedback, "#2|1|");
                char query[128] = "SELECT username, avatar FROM UserInfo WHERE id='";
                strcat(query, id);
                strcat(query, "';");
                char **name;
                if(sqlite3_get_table(db, query, &name, NULL, NULL, &sqlError) == 0){
                    strcat(feedback, name[2]);
                    strcat(feedback,"|");
                    strcat(feedback, name[3]);
                    strcat(feedback, "&");
                }else{
                    strcat(feedback, "NULL|4&");
                    printf("get username error: %s\n", sqlError);
                }
            }
            else
                strcpy(feedback, "#2|0|already logged in&");
        }
        else if(res == 0)
            strcpy(feedback, "#2|0|incorrect pw&");
    }
    send(fd, feedback, strlen(feedback), 0);

    return;
}


/*1 用户登录*/
void userRegister(char *data, int fd){
    //客户端传来：#1|avatar|name|password|safetypw&
    //printf("register.\n");
    int type = 0;
    char name[NAME_LEN] = {0}, pw[PW_LEN] = {0}, safety[SAFE_LEN] = {0}, avatar[AVATAR_LEN] = {0};

    //提取
    sscanf(data, "#%d|%[^|]|%[^|]|%[^|]|%[^&]", &type, avatar, name, pw, safety);

    //生成用户id,前面用0填充
    int tmp = ++cnt_users, v, i = 1;
    char id_str[ID_LEN] = "00000";
    while(tmp){
        v = tmp % 10;
        id_str[6 - (++i)] = v + '0';
        tmp = (tmp - v) / 10;
    }
    // printf("id = %s\n",id_str);

    //组织插入的sql语句
    char sql[1024] = "INSERT INTO userInfo(avatar,username,password,safetyPassword,id) VALUES(";
    strcat(sql, avatar);
    strcat(sql, ",'");
    strcat(sql, name);
    strcat(sql, "','");
    strcat(sql, pw);
    strcat(sql, "','");
    strcat(sql, safety);
    strcat(sql, "','");
    strcat(sql, id_str);
    strcat(sql, "');");
    // printf("%s\n", sql);

    //插入数据库，服务器回传：#1|0/1|xxxxx&
    char *sqlError = NULL;
    if (sqlite3_exec(db, sql, NULL, NULL, &sqlError) != 0){
        cnt_users--; //插入失败，计数减去
        strcpy(feedback,"#1|0|系统错误&");
        //printf("sql error: %s.\n", sqlError);
    } else{
        strcpy(feedback,"#1|1|");
        strcat(feedback,id_str);
        strcat(feedback,"&");
    }
    send(fd,feedback,strlen(feedback),0);

    return;
}


/*3 发送消息*/
void sendText(char *data, int fd){
    //客户端传来：#3|sender|receiver|text|time&
    //服务器返回：#3|time|sender id|sender username|text|sender avatar&
    int type = 0;
    char sender_id[ID_LEN] = {0}, receiver_id[ID_LEN] = {0}, content[CONTENT_LEN] = {0}, time[TIME_LEN] = {0};

    //提取消息内容
    sscanf(data, "#%d|%[^|]|%[^|]|%[^|]|%[^&]", &type, sender_id, receiver_id, content, time);

    //首先检查对方用户是否在线
    char sql[1024] = "SELECT COUNT(*) FROM Online WHERE id = '";
    strcat(sql,receiver_id);
    strcat(sql,"';");
    // printf("check online: %s\n",sql);
    char *sqlError = NULL;
    if(sqlite3_exec(db, sql, checkUser_Callback, 0, &sqlError) != 0){  //check失败
        printf("系统错误，请稍后重试！\n");
    }else{  //check成功
        //首先找出发送方信息
        char sender_username[NAME_LEN] = {0}, sender_avatar[AVATAR_LEN] = {0};
        char searchSenderInfo_sql[1024] = "SELECT username, avatar FROM UserInfo WHERE id = '";
        strcat(searchSenderInfo_sql, sender_id);
        strcat(searchSenderInfo_sql,"';");
        char **query_resultset = NULL;
        char *errmsg = NULL;
        int num_rows, num_cols, queryres;
        queryres = sqlite3_get_table(db, searchSenderInfo_sql, &query_resultset, &num_rows, &num_cols, &errmsg);
        if(queryres != 0){
            printf("get_table error : %s\n",errmsg);
            return;
        }else{
            if(num_rows != 1){
                printf("duplicate user!\n");
            }else{   //从结果表中取出
                strcpy(sender_username, query_resultset[2]);
                strcpy(sender_avatar, query_resultset[3]);
            }
            sqlite3_free_table(query_resultset);
        }

        //给接收端的反馈
        strcpy(feedback,"#3|");
        strcat(feedback,time);
        strcat(feedback,"|");
        strcat(feedback,sender_id);
        strcat(feedback,"|");
        strcat(feedback,sender_username);
        strcat(feedback,"|");
        strcat(feedback,content);
        strcat(feedback,"|");
        strcat(feedback,sender_avatar);
        strcat(feedback,"&");

        char msg_sql[1500] = "INSERT INTO ChatHistory(receiver, sender ,msg) VALUES('";
        char *err = NULL;
        strcat(msg_sql,receiver_id);
        strcat(msg_sql,"','");
        strcat(msg_sql,sender_id);
        strcat(msg_sql,"','");
        strcat(msg_sql,feedback);
        strcat(msg_sql,"');");
        if(sqlite3_exec(db,msg_sql, NULL, NULL, &err) != 0){
            printf("message loss: %s\n", err);
        }else{
            printf("save success.\n");
        }

        if(res == 1){  //如果用户在线，转发
            //首先找出接收者的connectfd
            int receiver_fd = findReceiverFd(receiver_id);
            send(receiver_fd,feedback,strlen(feedback),0);
        } else{
            printf("%d target users\n",res);
        }
    }

    return;
}


/*5 修改密码*/
void changePassword(char *data, int fd){
    //客户端传来：#5|id|safetyPassword|newPassword&
    //服务器返回：#5|0/1&
    int type = 0;
    char user_id[ID_LEN] = {0}, safe[SAFE_LEN] = {0}, new_pw[PW_LEN] = {0};

    //提取消息内容
    sscanf(data, "#%d|%[^|]|%[^|]|%[^&]", &type, user_id, safe, new_pw);

    //验证用户id和独立服务密码是否匹配
    char check_sql[512] = "SELECT COUNT(*) FROM UserInfo WHERE id='";
    strcat(check_sql, user_id);
    strcat(check_sql, "' AND safetyPassword='");
    strcat(check_sql, safe);
    strcat(check_sql, "';");
    // printf("check sql: %s\n",check_sql);
    char *sqlError = NULL;
    int exeres = sqlite3_exec(db, check_sql, checkUser_Callback, 0, &sqlError);
    if(exeres != 0){  //由于数据库错误验证失败
        send(fd,"#5|0",strlen("#5|0"),0);
        printf("系统错误，请稍后重试！\n");
    }else if(res == 1){  //验证成功且密码正确
        char update_sql[256] = "UPDATE UserInfo SET password='";
        strcat(update_sql, new_pw);
        strcat(update_sql, "' WHERE id='");
        strcat(update_sql, user_id);
        strcat(update_sql, "';");
        // printf("update sql: %s\n",update_sql);
        if(sqlite3_exec(db, update_sql, checkUser_Callback, 0, &sqlError) != 0){ //修改密码
            printf("change pw fail: %s\n",sqlError);
            send(fd,"#5|0",strlen("#5|0"),0);
        }else{   //修改成功
            send(fd,"#5|1",strlen("#5|1"),0);
        }
    }else{  //密码不正确
        printf("%d users!!\n",res);
        send(fd,"#5|0",strlen("#5|0"),0);
    }

    return;
}


/*6 传输文件*/
typedef struct transfile 
{
	char file_name[30];
	int file_len;
	char file_message[1024];
}File;


//#6|文件名|文件大小|文件&
void send_file(int confd, char* filepath,char *send_id, char *recv_id, char *time_str) 
{
    int flag = 1;
    File file;
    int count;
    char sendStr[2048] = "#6|";
	char charfilelen[6];
    memset(&file,0,sizeof(file));

    if (1 == flag) {
        flag = 0;
        FILE* filefd = fopen(filepath,"r");
        if (filefd == NULL) {
            perror("fopen");
            fclose(filefd);
            return;
        }   
        fseek(filefd,0,SEEK_END);
        file.file_len = ftell(filefd);
        fseek(filefd,0,SEEK_SET);
        printf("file_len :%d\n",file.file_len);
        strcpy(file.file_name,filepath);
        while ((count = fread(file.file_message,sizeof(char),1024,filefd) > 0) ) 
        {
            printf ("file_message lenth:%d\n",count);
            strcat(sendStr,file.file_name);
            strcat(sendStr,"|");
            sprintf(charfilelen,"%d",file.file_len);
            strcat(sendStr,charfilelen);
            strcat(sendStr,"|");
            strcat(sendStr,file.file_message);
            strcat(sendStr,"|");
            strcat(sendStr,send_id);
            strcat(sendStr,"|");
            strcat(sendStr,recv_id);
            strcat(sendStr,"|");
            strcat(sendStr,time_str);
            strcat(sendStr,"&");


            printf("%s\n",sendStr);
            if((count = write(confd, &sendStr, strlen(sendStr))) < 0)
            {
                perror("write");
                fclose(filefd);
                return ;
            }
            else
            {
                printf("%d\n",count);
            }
            memset(&file, 0, sizeof(file));
        }
    }  
}


/**
 * @brief recieve file from client
 * #6|文件名|文件大小|文件|发送人id|接收人id|时间&
 * @param char* data:
 * @param int confd Await connection to socket FD
 */
void recv_file(char* data,int confd, char* filename, char *send_id, char* recvid,char *time_str) {

	// recv block waiting , recvid[8] = {0}
	int count = 0, fileCurSize = 0, type = 0;
	File file;
    printf("data: %s\n",data);
	sscanf(data,"#%d|%[^|]|%d|%[^|]|%[^|]|%[^|]|%[^&]&",&type,file.file_name,&file.file_len,file.file_message,send_id,recvid,time_str);
	printf("%s %d %s\n",file.file_name,file.file_len,file.file_message);
    strcpy(filename, file.file_name);
	FILE* filefd = fopen(file.file_name,"w");
	if (filefd == NULL) {
		perror("fopen");
		fclose(filefd);
		return;
	}
	int len = fwrite(file.file_message,sizeof(char),file.file_len,filefd);
    fileCurSize += len;
    if (fileCurSize == file.file_len) {
        printf("recieve success\n");
    }
	fclose(filefd);
    return;
}


void transferFile(char *data, int fd){
    //客户端传来：#6|文件名|文件大小|文件|sender|receiver|time&
    //服务器返回：#6|文件名|文件大小|文件|sender|receiver|time&
    char filename[NAME_LEN] = {0}, receiver_id[ID_LEN] = {0},send_id[ID_LEN] = {0},time_str[TIME_LEN] = {0};
    recv_file(data, fd, filename,send_id, receiver_id,time_str);
    // printf("receive file: %s to %s\n",filename, receiver_id);

    //检查接收者是否在线
    char sql[128] = "SELECT COUNT(*) FROM Online WHERE id = '";
    strcat(sql,receiver_id);
    strcat(sql,"';");
    // printf("check recver sql: %s\n",sql);
    char *sqlError = NULL;
    if(sqlite3_exec(db, sql, checkUser_Callback, 0, &sqlError) != 0){  //check失败
        printf("系统错误，请稍后重试！\n");
    }else{  //check成功
        if(res == 0)  //不在线发不了
            return;
        else{  //在线则发送
            int receiver_fd = findReceiverFd(receiver_id);
            send_file(receiver_fd, filename,send_id,receiver_id,time_str);
        }
    }

    return;
}


/*7 添加好友*/
void addFriend(char *data, int fd){
    //客户端传来：#7|sender|receiver|s_username|_avatar&
    //服务器返回：#7|0/1|msg&
    int type = 0;
    char sender_id[ID_LEN] = {0}, receiver_id[ID_LEN] = {0}, s_username[NAME_LEN] = {0}, s_avatar[AVATAR_LEN] = {0};

    //提取消息内容
    sscanf(data, "#%d|%[^|]|%[^|]|%[^|]|%[^&]", &type, sender_id, receiver_id, s_username, s_avatar);

    char check_sql[128] = "SELECT COUNT(*) FROM UserInfo WHERE id = '";
    strcat(check_sql,receiver_id);
    strcat(check_sql,"';");
    char *sqlError = NULL;
    char **query_set = NULL;
    //首先检查要添加的用户是否存在
    if (sqlite3_exec(db, check_sql, checkUser_Callback, 0, &sqlError) != 0){
        strcpy(feedback,"#7|0|system error&");
    } else{
        if(res == 1){  //如果目标用户存在，检查两人是否已经是好友
            char check_sql_2[256] = "SELECT COUNT(*) FROM FriendRelation WHERE (id1='";
            strcat(check_sql_2, sender_id);
            strcat(check_sql_2, "' AND id2='");
            strcat(check_sql_2, receiver_id);
            strcat(check_sql_2, "') OR (id1='");
            strcat(check_sql_2, receiver_id);
            strcat(check_sql_2, "' AND id2='");
            strcat(check_sql_2, sender_id);
            strcat(check_sql_2, "');");
            // printf("%s\n",check_sql_2);
            if(sqlite3_exec(db, check_sql_2, checkUser_Callback, 0, &sqlError)!= 0){
                printf("%s\n", sqlError);
                strcpy(feedback,"#7|0|system error&");
            }else if(res == 0){  //不是好友：插入
                char addFriend_sql[256] = "INSERT INTO FriendRelation(id1, id2) VALUES('";
                strcat(addFriend_sql, sender_id);
                strcat(addFriend_sql, "','");
                strcat(addFriend_sql, receiver_id);
                strcat(addFriend_sql, "');");
                // printf("%s\n",addFriend_sql);
                if(sqlite3_exec(db, addFriend_sql, NULL, NULL, &sqlError) == 0){    //插入成功
                    // printf("add friend success\n");
                    strcpy(feedback, "#7|1|");
                    strcat(feedback, receiver_id);
                    strcat(feedback, "|");
                    char check_sql_3[128] = "SELECT username, avatar FROM UserInfo WHERE id='";
                    strcat(check_sql_3, receiver_id);
                    strcat(check_sql_3, "';");
                    //插入后给这个用户返回新好友的信息
                    if(sqlite3_get_table(db, check_sql_3, &query_set, NULL, NULL, &sqlError) == 0){
                        strcat(feedback, query_set[2]);
                        strcat(feedback, "|");
                        strcat(feedback, query_set[3]);
                        strcat(feedback, "&");    
                        
                        send(fd, feedback, strlen(feedback), 0);
                    }else{
                        printf("check new friend error: %s\n",feedback);
                    }

                    //给被添加的用户返回新好友信息
                    char check_sql_4[128] = "SELECT COUNT(*) FROM Online WHERE id = '";
                    strcat(check_sql_4, receiver_id);
                    strcat(check_sql_4, "';");
                    // printf("add friend: check receiver sql: %s\n",check_sql_4);
                    char *sqlError = NULL;
                    if(sqlite3_exec(db, check_sql_4, checkUser_Callback, 0, &sqlError) != 0){  //check失败
                        printf("系统错误，请稍后重试！\n");
                    }else if(res > 0){   //如果对方用户在线，反馈新好友信息;不在线就不发送，等下次登录时直接显示在好友列表里
                        //查询在线用户的fd
                        int receiver_fd = findReceiverFd(receiver_id);
                        strcpy(feedback, "#7|1|");
                        strcat(feedback, sender_id);
                        strcat(feedback, "|");
                        strcat(feedback, s_username);
                        strcat(feedback, "|");
                        strcat(feedback, s_avatar);
                        strcat(feedback, "&");
                        // printf("receiver:%s\n",feedback);
                        send(receiver_fd, feedback, strlen(feedback), 0);
                    }  
                } else{
                    printf("插入好友信息失败： %s\n",sqlError);
                    strcpy(feedback,"#7|0|system error!&");   
                    send(fd, feedback, strlen(feedback), 0);  
                }
            }else{  //已经是好友：不重复添加
                strcpy(feedback, "#7|0|已经是好友&");
                send(fd, feedback, strlen(feedback), 0);
            }
        }
    }   

    return;
}


/*8 用户上线后获取消息列表*/
void checkMessages(char *data, int fd){
    //服务端反馈消息状态：#8|0/1/2|查看人id（接收者）|被查看人id（发送者）&
    int type = 0, msg_state = 0;
    char sender_id[ID_LEN] = {0}, receiver_id[ID_LEN] = {0}, content[CONTENT_LEN] = {0}, time[TIME_LEN] = {0};

    //提取消息内容
    sscanf(data, "#%d|%d|%[^|]|%[^&]", &type, &msg_state, receiver_id, sender_id);

    if(msg_state == 0)  //0代表未查看信息，不做任何操作
        return;

    char sql[1200] = {0};
    char *errormsg = NULL;
    //1代表接收方当前正在聊天窗口，设为已读
    //2代表接收方当前不在此聊天窗口，现在要点开，返回这两个人所有的消息记录并设为已读
    strcpy(sql, "Update ChatHistory SET read=1 WHERE sender='");
    strcat(sql, sender_id);
    strcat(sql, "' AND receiver='");        
    strcat(sql, receiver_id);
    strcat(sql, "';");
    printf("%s\n",sql);
    if(sqlite3_exec(db, sql, NULL, NULL, &errormsg) != 0){
        printf("%s\n", errormsg);
        return;
    }
    if(msg_state == 2)   //返回聊天记录
        getChatHistory(receiver_id, sender_id, fd);

    return;
}


/*9 退出登录*/
void logOut(char *data, int fd){
    //服务端反馈消息状态：#9|id&
    //没有反馈消息
    int type = 0;
    char id[ID_LEN] = {0};

    //提取消息内容
    sscanf(data, "#%d|%[^&]", &type, id);

    char sql[64] = "DELETE FROM Online WHERE id='";
    strcat(sql, id);
    strcat(sql, "';");
    char *err;

    if(sqlite3_exec(db, sql, NULL, NULL, &err) != 0){
        printf("log out fail.\n");
    }else{
        printf("log out success.\n");
    }

    return;
}


/*处理接收数据的接口函数*/
void handData(char *data, int fd){
    if (data[0] != '#' || data[strlen(data) - 1] != '&')
        return;

    switch (data[1]){
        case '1':  //注册业务
            userRegister(data, fd);
            break;
        case '2':  //登录
            logIn(data, fd);
            break;
        case '3':  //发消息
            sendText(data,fd);
            break;
        case '4':  //好友与消息列表
            getFriendsList(data, fd);
            break;
        case '5':  //修改密码
            changePassword(data, fd);
            break;
        case '6':  //传输文件
            transferFile(data,fd);
            break;
        case '7':  //添加好友
            addFriend(data, fd);
            break;
        case '8':  //更新在线用户的消息状态
            checkMessages(data, fd);
            break;
        case '9':  //退出登录
            logOut(data, fd);
            break;
        default:
            printf("default\n");
            return;
    }
    return;
}


int main(){
    //打开数据库
    int openDB = sqlite3_open("ServerData.db", &db); //句柄
    if (openDB != 0){
        printf("Open DB error code = %d\n", openDB);
        return -1;
    }

    //数出当前有多少用户，用于后续生成id
    char *msg = NULL;
    if(sqlite3_exec(db, "SELECT COUNT(*) AS count FROM userInfo;", checkTable_Callback, 0, &msg) != 0){
        printf("%s\n",msg);
        return -1;
    }

    //每次开启服务器时清空在线用户表
    if(sqlite3_exec(db, "DELETE FROM Online;", NULL, NULL, &msg) != 0){
        printf("%s\n",msg);
        return -1;
    }

    //监听
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd == -1){
        printf("socket error\n");
        return -1;
    }
    struct sockaddr_in myaddr;
    memset(&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET; // IPv4
    myaddr.sin_port = htons(8899); // 注释: 端口号
    myaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listenfd, (struct sockaddr *)&myaddr, sizeof(myaddr)) == -1){
        printf("bind error\n");
        return -1;
    }
    listen(listenfd, 10);  //监听等待队列大小为10

    // IO复用
    fd_set readset, allset; // allset: backup
    int maxfd = 0, connectfd = 0;
    int cnt = 0, i = 0;
    int clientfd[1024] = {0}; // 1024: max number of clients
    FD_ZERO(&allset);
    FD_SET(listenfd, &allset);
    if (maxfd < listenfd)
        maxfd = listenfd;

    while (1){
        readset = allset;
        printf("select\n");
        // 等待0.01秒
        usleep(10000);
        

        //fd三状态：可读、可写、异常。把fd放到第几个参数就是监测哪个状态；NULL：阻塞
        int num_ready = select(maxfd + 1, &readset, NULL, NULL, NULL);
        if (num_ready == -1){
            printf("select error\n");
            continue;
        }
        if (FD_ISSET(listenfd, &readset)){
            connectfd = accept(listenfd, NULL, NULL);  //处理客户端的新连接，返回真正进行通信的句柄（后两个获取客户端地址信息）
            printf("new connection\n");
            if (connectfd == -1){
                printf("accept error\n");
                return -1;
            }
            FD_SET(connectfd, &allset);
            if (connectfd > maxfd)
                maxfd = connectfd;  //算数值最大的fd
            
            for (i = 0; i < cnt; i++){ 
                if(clientfd[i] == 0){
                    clientfd[i] = connectfd;
                    break;
                }
            }
            if (i == cnt){
                if (cnt >= 1024)
                    printf("max\n");
                else
                    clientfd[cnt++] = connectfd;
            }
            if (--num_ready == 0)
                continue;
        }

        for (i = 0; i < cnt; i++){   //遍历每一个连接的用户
            connectfd = clientfd[i];
            if (FD_ISSET(connectfd, &readset)){ 
                // printf("i-th : %d\n", i);
                char buf[1024] = {0};
                int recv_ret = recv(connectfd, buf, sizeof(buf), 0);  //第一个参数必须是连接描述符（accept返回的地址）；第三个：最多接收多少数据
                // printf("recv_ret = %d\n", recv_ret);
                if (recv_ret <= 0){
                    FD_CLR(connectfd, &allset);
                    clientfd[i] = 0; // useless space
                    continue;
                }
                printf("recv = %s\n", buf);
                handData(buf, connectfd);

                if (--num_ready == 0) // no need to go over all the 1024
                    break;
            }
        }
    }

    sqlite3_close(db); //断开数据库连接
    return 0;
}
