package com.spring.biz.user.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import com.spring.biz.user.UserVO;

@Repository
public class UserDAOSpring {
	
	@Autowired
	private JdbcTemplate jdbcTemplate;
	
	private final String USER_INSERT = 
			"insert into users values (?, ?, ?, ?)";
	private final String USER_UPDATE = 
			"update users set name=?, role=? where id=?";
	private final String USER_DELETE = 
			"delete from users where id=?";
	private final String USER_GET = 
			"select * from users where id=?";
	private final String USER_LIST = 
			"select * from users order by id desc";
	private final String USER_CHECK =
			"select * from users where id=? and password=?";
	
	public void insertUser(UserVO vo) {
		jdbcTemplate.update(USER_INSERT, vo.getId(), vo.getPassword(), vo.getName(), vo.getRole());
	}
	
	public void updateUser(UserVO vo) {
		jdbcTemplate.update(USER_UPDATE, vo.getName(), vo.getRole(), vo.getId());
	}
	
	public void deleteUser(UserVO vo) {
		jdbcTemplate.update(USER_DELETE, vo.getId());
	}
	public UserVO getUser(UserVO vo) {
		Object [] args = {vo.getId()};
		return jdbcTemplate.queryForObject(USER_GET, args, new UserMapper());
	}
	public List<UserVO> getUserList(){
		return jdbcTemplate.query(USER_LIST, new UserMapper());
	}
	public UserVO checkUserLogin(UserVO vo) {
		Object [] args = {vo.getId(), vo.getPassword()};
		return jdbcTemplate.queryForObject(USER_CHECK, args, new UserMapper());
	}
}
