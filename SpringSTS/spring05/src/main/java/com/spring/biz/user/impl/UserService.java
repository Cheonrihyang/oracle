package com.spring.biz.user.impl;

import java.util.List;

import com.spring.biz.user.UserVO;

public interface UserService {
	
	public void insertUser(UserVO vo);
	public void updateUser(UserVO vo);
	public void deleteUser(UserVO vo);
	public UserVO getUser(UserVO vo);
	public List<UserVO> getUserList();
	public UserVO checkuserLogin(UserVO vo);
}
